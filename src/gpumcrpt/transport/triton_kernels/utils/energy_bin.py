from __future__ import annotations

import triton
import triton.language as tl

from gpumcrpt.transport.triton_kernels.utils.gpu_math import fast_log_approx


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
    key=['n'],
    warmup=10,
    rep=20,
)
@triton.jit
def compute_energy_bin_kernel(
    E_MeV_ptr,
    out_ebin_ptr,
    n: tl.constexpr,
    common_log_E_min: tl.constexpr,
    common_log_step_inv: tl.constexpr,
    NB: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NUM_WARPS: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    pid = tl.program_id(0)
    i = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    m = i < n

    E = tl.load(E_MeV_ptr + i, mask=m, other=0.0).to(tl.float32)
    Epos = tl.maximum(E, 1e-30)
    lnE = fast_log_approx(Epos)
    t = (lnE - common_log_E_min) * common_log_step_inv
    ebin = t.to(tl.int32)
    ebin = tl.maximum(0, tl.minimum(NB - 1, ebin))
    tl.store(out_ebin_ptr + i, ebin, mask=m)
