from __future__ import annotations

import triton
import triton.language as tl


@triton.jit
def reduce_bins_hash_active_kernel_r1(
    active_bins_ptr,      # int32 [A]
    bin_offsets_ptr,      # int32 [n_bins]
    bin_counts_ptr,       # int32 [n_bins]
    in_lin_ptr, in_val_ptr,
    edep_ptr,
    probe_fail_ptr,       # int32 [A] optional (can be nullptr-like via mask on host)
    first_probe_miss_ptr, # int32 [A] optional
    A: tl.constexpr,
    H: tl.constexpr,          # power-of-two, == BLOCK
    PROBES: tl.constexpr,
):
    """
    Fast R1 reducer:
      - H == processing block size.
      - local hash table size H in registers.
      - bounded probing.
    Diagnostics (per active bin):
      - first_probe_miss: number of inserts that did not match on first slot
      - probe_fail: number of inserts that did not succeed after all probes (should be ~0)
    """
    pid = tl.program_id(0)
    if pid >= A:
        return

    b = tl.load(active_bins_ptr + pid).to(tl.int32)
    start = tl.load(bin_offsets_ptr + b).to(tl.int32)
    count = tl.load(bin_counts_ptr + b).to(tl.int32)
    end = start + count

    keys = tl.full((H,), -1, tl.int32)
    acc = tl.zeros((H,), tl.float32)

    miss0 = tl.zeros((), tl.int32)
    fail = tl.zeros((), tl.int32)

    i = start
    while i < end:
        offs = i + tl.arange(0, H)
        m = offs < end

        lin = tl.load(in_lin_ptr + offs, mask=m, other=-1).to(tl.int32)
        val = tl.load(in_val_ptr + offs, mask=m, other=0.0).to(tl.float32)

        good = lin >= 0
        lin = tl.where(good, lin, -1)
        val = tl.where(good, val, 0.0)

        s = lin & (H - 1)
        done = lin == -1

        # Track first-slot success (before any probing):
        k0 = tl.load(keys + s, mask=True, other=-1)
        first_ok = (~done) & ((k0 == -1) | (k0 == lin))
        miss0 += tl.sum((~done & ~first_ok).to(tl.int32), axis=0)

        for _ in range(PROBES):
            k = tl.load(keys + s, mask=True, other=-1)
            empty = k == -1
            same = k == lin
            can = (~done) & (empty | same)

            # claim empty slots: set key to lin
            keys = tl.where(can & empty, lin, keys)

            # add to corresponding slot s
            idx = tl.arange(0, H)
            addv = tl.where(can, val, 0.0)
            acc = acc + tl.where(idx == s, addv, 0.0)

            done = done | can
            s = (s + 1) & (H - 1)

        fail += tl.sum((~done).to(tl.int32), axis=0)

        i += H

    # flush
    idx = tl.arange(0, H)
    k = tl.load(keys + idx, mask=True, other=-1).to(tl.int32)
    v = tl.load(acc + idx, mask=True, other=0.0).to(tl.float32)
    good = k >= 0
    tl.atomic_add(edep_ptr + k, v, mask=good)

    # diagnostics (optional: pointers can be None on host by not allocating, but Triton needs tensors;
    # host should pass dummy 1-element if disabled, and gate with mask)
    tl.store(first_probe_miss_ptr + pid, miss0, mask=True)
    tl.store(probe_fail_ptr + pid, fail, mask=True)