from __future__ import annotations

import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128, 'NUM_WARPS': 4, 'NUM_STAGES': 4}),
        triton.Config({'BLOCK_SIZE': 128, 'NUM_WARPS': 4, 'NUM_STAGES': 3}),
        triton.Config({'BLOCK_SIZE': 256, 'NUM_WARPS': 4, 'NUM_STAGES': 4}),
        triton.Config({'BLOCK_SIZE': 256, 'NUM_WARPS': 4, 'NUM_STAGES': 3}),
        triton.Config({'BLOCK_SIZE': 256, 'NUM_WARPS': 8, 'NUM_STAGES': 2}),
        triton.Config({'BLOCK_SIZE': 512, 'NUM_WARPS': 4, 'NUM_STAGES': 3}),
        triton.Config({'BLOCK_SIZE': 512, 'NUM_WARPS': 8, 'NUM_STAGES': 2}),
        triton.Config({'BLOCK_SIZE': 512, 'NUM_WARPS': 8, 'NUM_STAGES': 3}),
        triton.Config({'BLOCK_SIZE': 1024, 'NUM_WARPS': 8, 'NUM_STAGES': 2}),
        triton.Config({'BLOCK_SIZE': 1024, 'NUM_WARPS': 16, 'NUM_STAGES': 1}),
        triton.Config({'BLOCK_SIZE': 1024, 'NUM_WARPS': 16, 'NUM_STAGES': 2}),
    ],
    key=['Ns'],
    warmup=10,
    rep=20,
)
@triton.jit
def append_vacancies_bank_soa_kernel(
    src_x_ptr, src_y_ptr, src_z_ptr,
    src_atom_Z_ptr,
    src_shell_idx_ptr,
    src_w_ptr,
    src_id_ptr,

    dst_x_ptr, dst_y_ptr, dst_z_ptr,
    dst_atom_Z_ptr,
    dst_shell_idx_ptr,
    dst_w_ptr,
    dst_id_ptr,
    dst_status_ptr,

    global_counters_ptr,
    VAC_COUNT_IDX: tl.constexpr,

    Ns: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NUM_WARPS: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    pid = tl.program_id(0)
    i = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    m = i < Ns

    x = tl.load(src_x_ptr + i, mask=m, other=0.0).to(tl.float32)
    y = tl.load(src_y_ptr + i, mask=m, other=0.0).to(tl.float32)
    z = tl.load(src_z_ptr + i, mask=m, other=0.0).to(tl.float32)
    Z = tl.load(src_atom_Z_ptr + i, mask=m, other=0).to(tl.int32)
    shell = tl.load(src_shell_idx_ptr + i, mask=m, other=0).to(tl.int32)
    w = tl.load(src_w_ptr + i, mask=m, other=0.0).to(tl.float32)
    vac_id = tl.load(src_id_ptr + i, mask=m, other=0).to(tl.int64)

    has = (Z > 0) | (shell >= 0)

    dst_idx = tl.atomic_add(global_counters_ptr + VAC_COUNT_IDX, 1, mask=m & has).to(tl.int32)

    tl.store(dst_x_ptr + dst_idx, x, mask=m & has)
    tl.store(dst_y_ptr + dst_idx, y, mask=m & has)
    tl.store(dst_z_ptr + dst_idx, z, mask=m & has)
    tl.store(dst_atom_Z_ptr + dst_idx, Z, mask=m & has)
    tl.store(dst_shell_idx_ptr + dst_idx, shell, mask=m & has)
    tl.store(dst_w_ptr + dst_idx, w, mask=m & has)
    tl.store(dst_id_ptr + dst_idx, vac_id, mask=m & has)
    tl.store(dst_status_ptr + dst_idx, 1, mask=m & has)
