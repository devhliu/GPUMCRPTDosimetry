from __future__ import annotations

import triton
import triton.language as tl


@triton.jit
def build_active_bins_padded_kernel(
    mask_ptr, prefix_ptr,
    out_active_bins_ptr,
    n: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """
    Build a padded active bins list:
      - out_active_bins is initialized to -1 by host
      - for each i with mask[i]=1, write i into out_active_bins[prefix[i]]
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < n

    mask = tl.load(mask_ptr + offs, mask=m, other=0).to(tl.int1)
    pref = tl.load(prefix_ptr + offs, mask=m, other=0).to(tl.int32)

    # write i at its compact position
    # (only for active slots)
    tl.store(out_active_bins_ptr + pref, offs.to(tl.int32), mask=m & mask)


@triton.jit
def reduce_bins_hash_active_padded_kernel_r1(
    active_bins_padded_ptr,  # int32[n_bins], padded with -1
    bin_offsets_ptr, bin_counts_ptr,
    in_lin_ptr, in_val_ptr,
    edep_ptr,
    n_bins: tl.constexpr,
    H: tl.constexpr,          # == BLOCK
    PROBES: tl.constexpr,
):
    """
    One program per possible active-entry slot (0..n_bins-1).
    Reads bin = active_bins_padded[pid]. If -1 => return.
    """
    pid = tl.program_id(0)
    if pid >= n_bins:
        return

    b = tl.load(active_bins_padded_ptr + pid).to(tl.int32)
    if b < 0:
        return

    start = tl.load(bin_offsets_ptr + b).to(tl.int32)
    count = tl.load(bin_counts_ptr + b).to(tl.int32)
    end = start + count

    keys = tl.full((H,), -1, tl.int32)
    acc = tl.zeros((H,), tl.float32)

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

        for _ in range(PROBES):
            k = tl.load(keys + s, mask=True, other=-1)
            empty = k == -1
            same = k == lin
            can = (~done) & (empty | same)

            keys = tl.where(can & empty, lin, keys)

            idx = tl.arange(0, H)
            addv = tl.where(can, val, 0.0)
            acc = acc + tl.where(idx == s, addv, 0.0)

            done = done | can
            s = (s + 1) & (H - 1)

        i += H

    idx = tl.arange(0, H)
    k = tl.load(keys + idx, mask=True, other=-1).to(tl.int32)
    v = tl.load(acc + idx, mask=True, other=0.0).to(tl.float32)
    good = k >= 0
    tl.atomic_add(edep_ptr + k, v, mask=good)