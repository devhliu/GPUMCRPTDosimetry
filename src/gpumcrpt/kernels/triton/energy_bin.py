from __future__ import annotations

import triton
import triton.language as tl


@triton.jit
def compute_ebin_log_uniform_kernel(
    E_MeV_ptr,
    out_ebin_ptr,
    n: tl.constexpr,
    common_log_E_min: tl.constexpr,       # ln(E_min), E in MeV
    common_log_step_inv: tl.constexpr,    # 1 / d(lnE)
    NB: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    i = pid * BLOCK + tl.arange(0, BLOCK)
    m = i < n

    E = tl.load(E_MeV_ptr + i, mask=m, other=0.0).to(tl.float32)
    Epos = tl.maximum(E, 1e-30)
    lnE = tl.log(Epos)
    t = (lnE - common_log_E_min) * common_log_step_inv
    ebin = t.to(tl.int32)
    ebin = tl.maximum(0, tl.minimum(NB - 1, ebin))
    tl.store(out_ebin_ptr + i, ebin, mask=m)