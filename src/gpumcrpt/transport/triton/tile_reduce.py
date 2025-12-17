from __future__ import annotations

import triton
import triton.language as tl


@triton.jit
def tile_hash_reduce_kernel(
    bin_offsets_ptr, bin_counts_ptr,
    lin_ptr, val_ptr,
    edep_ptr,
    n_bins: tl.constexpr,
    H: tl.constexpr,
    PROBES: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """
    One program per tile bin.
    - Reads that tile's contiguous (lin,val) segment.
    - Accumulates in a small hash table of size H (in registers).
    - Flushes to edep via atomic adds for occupied slots.

    Notes:
    - This is a simplified kernel; tune BLOCK/H/PROBES for performance.
    """
    tile = tl.program_id(0)
    if tile >= n_bins:
        return

    start = tl.load(bin_offsets_ptr + tile).to(tl.int32)
    count = tl.load(bin_counts_ptr + tile).to(tl.int32)
    end = start + count

    # local hash arrays
    keys = tl.full((H,), -1, tl.int32)
    acc = tl.zeros((H,), tl.float32)

    # iterate events in chunks
    # bounded loop: max iters = ceil(count/BLOCK) but count varies.
    # For GPU-friendliness: limit to MAX_CHUNKS at capture-time; outside graphs ok.
    # Here we allow while; this kernel is intended outside graphs initially.
    i = start
    while i < end:
        offs = i + tl.arange(0, BLOCK)
        m = offs < end
        lin = tl.load(lin_ptr + offs, mask=m, other=-1).to(tl.int32)
        val = tl.load(val_ptr + offs, mask=m, other=0.0).to(tl.float32)

        # insert each element using vectorized probes
        h0 = lin & (H - 1)
        h = h0
        for _ in range(PROBES):
            k = tl.load(keys + h, mask=True, other=-1)
            empty = k == -1
            same = k == lin
            can_write = empty | same

            # if empty, claim slot
            do_claim = empty & (lin >= 0)
            keys = tl.where(do_claim, tl.multiple_of(lin, 1), keys)  # no-op hint

            # update: if can_write, add val
            acc_add = tl.where(can_write, val, 0.0)
            acc = acc + tl.where(tl.arange(0, H) == h, acc_add, 0.0)

            # for those not written, probe next
            h = (h + 1) & (H - 1)

        i += BLOCK

    # flush occupied slots
    occ = keys != -1
    # flush in blocks to avoid huge single instruction
    for j in range(0, H, BLOCK):
        jj = j + tl.arange(0, BLOCK)
        m = jj < H
        k = tl.load(keys + jj, mask=m, other=-1).to(tl.int32)
        v = tl.load(acc + jj, mask=m, other=0.0).to(tl.float32)
        good = k >= 0
        tl.atomic_add(edep_ptr + k, v, mask=good)