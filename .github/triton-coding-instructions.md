# Triton 3.5.1 Coding Instructions

## Overview

This document provides comprehensive coding guidelines for using Triton 3.5.1 in GPU-accelerated Monte Carlo dosimetry simulations. These instructions are based on the official Triton documentation and optimized for the specific needs of this project.

## Table of Contents

1. [Basic Setup and Imports](#basic-setup-and-imports)
2. [Kernel Structure and Decorators](#kernel-structure-and-decorators)
3. [Memory Access Patterns](#memory-access-patterns)
4. [Data Types and Operations](#data-types-and-operations)
5. [Random Number Generation](#random-number-generation)
6. [Performance Optimization](#performance-optimization)
7. [Testing and Debugging](#testing-and-debugging)
8. [Common Patterns for Monte Carlo Simulations](#common-patterns-for-monte-carlo-simulations)
9. [Best Practices](#best-practices)

## Basic Setup and Imports

### Required Imports

```python
from __future__ import annotations
import torch
import triton
import triton.language as tl
```

### Version Compatibility

Ensure Triton version 3.5.1 is specified in dependencies:
```toml
# pyproject.toml
dependencies = [
    "triton>=3.5.1",
]
```

## Kernel Structure and Decorators

### Basic Kernel Template

```python
@triton.jit
def kernel_name(
    # Pointers to input/output tensors
    input_ptr,
    output_ptr,
    
    # Problem dimensions (must be tl.constexpr)
    n_elements: tl.constexpr,
    
    # Configuration parameters (must be tl.constexpr)
    BLOCK_SIZE: tl.constexpr,
    
    # Optional: strides for multi-dimensional arrays
    stride_0: tl.constexpr,
    stride_1: tl.constexpr,
):
    # Program ID and offsets
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # Mask for boundary checking
    mask = offs < n_elements
    
    # Load data
    data = tl.load(input_ptr + offs, mask=mask)
    
    # Computation
    result = data * 2.0
    
    # Store result
    tl.store(output_ptr + offs, result, mask=mask)
```

### Autotuning Configuration

```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 64, 'NUM_WARPS': 2}, num_stages=4),
        triton.Config({'BLOCK_SIZE': 128, 'NUM_WARPS': 4}, num_stages=3),
        triton.Config({'BLOCK_SIZE': 256, 'NUM_WARPS': 4}, num_stages=2),
        triton.Config({'BLOCK_SIZE': 512, 'NUM_WARPS': 8}, num_stages=1),
    ],
    key=['n_elements'],  # Tune based on problem size
)
@triton.jit
def optimized_kernel(...):
    # Kernel implementation
    pass
```

## Memory Access Patterns

### Modern Block Pointers (Recommended)

```python
@triton.jit
def modern_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    # Create block pointers
    a_block_ptr = tl.make_block_ptr(
        base=A_ptr,
        shape=(M, K),
        strides=(stride_am, stride_ak),
        offsets=(pid_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_K),
        order=(1, 0)
    )
    
    # Load with boundary checking
    a = tl.load(a_block_ptr, boundary_check=(0, 1))
    
    # Advance pointer
    a_block_ptr = tl.advance(a_block_ptr, (0, k))
```

### Cache Control

```python
# Cache Global - for streaming data
x = tl.load(ptr + offs, mask=mask, cache_modifier=".cg")

# Cache All - for heavily reused data
weights = tl.load(weight_ptr, mask=mask, cache_modifier=".ca")
```

## Data Types and Operations

### Supported Data Types

```python
# Integer types
i32 = tl.int32
i64 = tl.int64

# Floating point types
f32 = tl.float32
f64 = tl.float64

# Type conversions
data_f32 = data.to(tl.float32)
data_i32 = data.to(tl.int32)
```

### Mathematical Operations

```python
# Basic arithmetic
a = x + y
b = x - y
c = x * y
d = x / y

# Mathematical functions
log_val = tl.log(x)
exp_val = tl.exp(x)
sqrt_val = tl.sqrt(x)

# Trigonometric functions
sin_val = tl.sin(x)
cos_val = tl.cos(x)

# Comparison operations
mask = x > y
mask = x == y
```

### Atomic Operations

```python
# Atomic add
tl.atomic_add(output_ptr + index, value)

# Atomic max
tl.atomic_max(output_ptr + index, value)

# Atomic min
tl.atomic_min(output_ptr + index, value)
```

## Random Number Generation

### Philox RNG Implementation

```python
@triton.jit
def _mulhi_u32(a: tl.tensor, b: tl.tensor) -> tl.tensor:
    return ((a.to(tl.uint64) * b.to(tl.uint64)) >> 32).to(tl.uint32)

@triton.jit
def philox4x32_round(c0, c1, c2, c3, k0, k1):
    """Single round of Philox 4x32 RNG"""
    h0 = _mulhi_u32(0xD2511F53, c0)
    h1 = _mulhi_u32(0xCD9E8D57, c2)
    
    c0 = h0 ^ c2 ^ k0
    c2 = h1 ^ c0 ^ k1
    c1 = 0xCD9E8D57 * c1 + 0x9E3779B9
    c3 = 0xD2511F53 * c3 + 0xBB67AE85
    
    return c0, c1, c2, c3

@triton.jit
def rand_uniform_u01_philox(
    key0: tl.tensor, key1: tl.tensor,
    ctr0: tl.tensor, ctr1: tl.tensor, ctr2: tl.tensor, ctr3: tl.tensor,
    rounds: tl.constexpr = 10
) -> tl.tensor:
    """Generate uniform random numbers using Philox RNG"""
    c0, c1, c2, c3 = ctr0, ctr1, ctr2, ctr3
    
    for _ in range(rounds):
        c0, c1, c2, c3 = philox4x32_round(c0, c1, c2, c3, key0, key1)
    
    # Convert to float in [0, 1)
    u = c0.to(tl.float32) / 4294967296.0
    return u
```

## Performance Optimization

### Compiler Hints

```python
# Multiple of hint for vectorization
stride = tl.multiple_of(stride, 16)

# Max contiguous hint
block_size = tl.max_contiguous(block_size, 16)
```

### Grid Swizzling for L2 Cache

```python
@triton.jit
def optimized_kernel(...):
    # Grid swizzling for better cache locality
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
```

### Pipeline Stages

```python
@triton.jit(num_stages=3)  # Enable software pipelining
def pipelined_kernel(...):
    # Kernel implementation with multiple stages
    pass
```

## Testing and Debugging

### Basic Testing Pattern

```python
import torch
import triton
import triton.testing as tt


def test_kernel():
    # Create test data
    n = 1024
    x = torch.randn(n, device='cuda', dtype=torch.float32)
    y = torch.empty_like(x)
    
    # Define grid
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)
    
    # Launch kernel
    kernel[grid](x, y, n, BLOCK_SIZE=256)
    
    # Verify results
    expected = x * 2.0
    torch.testing.assert_close(y, expected)


# Performance benchmarking
@tt.perf_report(
    tt.Benchmark(
        x_names=['n'],
        x_vals=[2**i for i in range(10, 20)],
        line_arg='provider',
        line_vals=['triton', 'torch'],
        line_names=['Triton', 'Torch'],
        styles=[('blue', '-'), ('green', '-')],
        ylabel='GB/s',
        plot_name='kernel-performance',
        args={},
    )
)
def benchmark(n, provider):
    # Benchmark implementation
    pass
```

### Debugging Techniques

```python
# Use torch.compile for debugging CPU execution
@torch.compile
def debug_kernel():
    # Debug implementation
    pass

# Print debugging (limited support)
# Note: Use sparingly as it impacts performance
if pid == 0 and offs[0] == 0:
    # Limited print support
    pass
```

## Common Patterns for Monte Carlo Simulations

### Particle Transport Kernel

```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256, 'NUM_WARPS': 4}, num_stages=3),
    ],
    key=['n_particles']
)
@triton.jit
def particle_transport_kernel(
    # Particle properties (Structure of Arrays)
    x_ptr, y_ptr, z_ptr,
    dx_ptr, dy_ptr, dz_ptr,
    energy_ptr, weight_ptr,
    
    # RNG states
    rng_key0_ptr, rng_key1_ptr,
    rng_ctr0_ptr, rng_ctr1_ptr, rng_ctr2_ptr, rng_ctr3_ptr,
    
    # Geometry and materials
    material_id_ptr,
    density_ptr,
    
    # Output: energy deposition
    edep_ptr,
    
    # Problem dimensions
    n_particles: tl.constexpr,
    grid_x: tl.constexpr, grid_y: tl.constexpr, grid_z: tl.constexpr,
    
    # Configuration
    BLOCK_SIZE: tl.constexpr,
    photon_cut_MeV: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_particles
    
    # Load particle data
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    z = tl.load(z_ptr + offs, mask=mask)
    energy = tl.load(energy_ptr + offs, mask=mask)
    
    # Load RNG state
    key0 = tl.load(rng_key0_ptr + offs, mask=mask)
    ctr0 = tl.load(rng_ctr0_ptr + offs, mask=mask)
    
    # Generate random numbers
    rand = rand_uniform_u01_philox(key0, key1, ctr0, ctr1, ctr2, ctr3)
    
    # Particle transport logic
    # ...
    
    # Atomic energy deposition
    voxel_index = get_voxel_index(x, y, z, grid_x, grid_y, grid_z)
    tl.atomic_add(edep_ptr + voxel_index, energy_deposited)
```

### Bank Management Pattern

```python
@triton.jit
def bank_append_kernel(
    # Input particles to append
    src_x_ptr, src_y_ptr, src_z_ptr,
    src_energy_ptr, src_weight_ptr,
    
    # Destination bank
    dst_x_ptr, dst_y_ptr, dst_z_ptr,
    dst_energy_ptr, dst_weight_ptr,
    
    # Bank counters (atomic)
    counter_ptr,
    
    n_src: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_src
    
    # Load source data
    x = tl.load(src_x_ptr + offs, mask=mask)
    energy = tl.load(src_energy_ptr + offs, mask=mask)
    
    # Atomic increment to get destination index
    dst_index = tl.atomic_add(counter_ptr, 1, mask=mask)
    
    # Store to destination
    tl.store(dst_x_ptr + dst_index, x, mask=mask)
    tl.store(dst_energy_ptr + dst_index, energy, mask=mask)
```

## Best Practices

### Code Organization

1. **Keep kernels focused**: Each kernel should perform a single, well-defined operation
2. **Use meaningful names**: Clear variable and function names improve readability
3. **Document constraints**: Comment on any limitations or assumptions
4. **Follow project conventions**: Maintain consistency with existing codebase

### Performance Guidelines

1. **Always use autotuning**: Never hardcode block sizes
2. **Prefer block pointers**: Use modern memory access patterns
3. **Minimize global memory access**: Use registers and local computation when possible
4. **Optimize for your target architecture**: RTX A4000 (Ampere) optimizations are included

### Compatibility Notes for Triton 3.5.1

**Supported Features:**
- Block pointers (`tl.make_block_ptr`, `tl.advance`)
- Cache modifiers (`.cg`, `.ca`)
- Autotuning
- Atomic operations
- Most mathematical functions

**Unsupported Features (Comment Out):**
- `tl.prefetch`
- `tl.static_shared_memory`
- `tl.barrier`
- Module imports inside `@triton.jit` functions

### Error Handling

```python
# Always validate inputs before kernel launch
def safe_kernel_launch(kernel, grid, *args, **kwargs):
    # Check tensor properties
    for arg in args:
        if isinstance(arg, torch.Tensor):
            assert arg.is_cuda, "All tensors must be on CUDA device"
            assert arg.is_contiguous(), "Tensors must be contiguous"
    
    # Launch kernel
    kernel[grid](*args, **kwargs)
```

## Example: Complete Monte Carlo Step

```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256, 'NUM_WARPS': 4}, num_stages=3),
    ],
    key=['n_photons']
)
@triton.jit
def monte_carlo_photon_step(
    # Photon bank (SoA layout)
    photon_x_ptr, photon_y_ptr, photon_z_ptr,
    photon_dx_ptr, photon_dy_ptr, photon_dz_ptr,
    photon_energy_ptr, photon_weight_ptr,
    photon_status_ptr,
    
    # RNG states
    photon_key0_ptr, photon_key1_ptr,
    photon_ctr0_ptr, photon_ctr1_ptr, photon_ctr2_ptr, photon_ctr3_ptr,
    
    # Output banks
    electron_x_ptr, electron_y_ptr, electron_z_ptr,
    electron_energy_ptr, electron_weight_ptr,
    electron_counter_ptr,
    
    # Energy deposition
    edep_ptr,
    
    # Problem parameters
    n_photons: tl.constexpr,
    grid_dims: tl.constexpr,
    material_table_ptr,
    cross_section_ptr,
    BLOCK_SIZE: tl.constexpr,
    photon_cutoff: tl.constexpr,
):
    """Single Monte Carlo step for photon transport"""
    
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_photons
    
    # Load photon data
    x = tl.load(photon_x_ptr + offs, mask=mask)
    y = tl.load(photon_y_ptr + offs, mask=mask)
    energy = tl.load(photon_energy_ptr + offs, mask=mask)
    
    # Only process active photons
    active_mask = mask & (energy > photon_cutoff)
    
    if tl.sum(active_mask) > 0:
        # Photon flight and interaction
        # ... implementation details ...
        
        # Energy deposition
        voxel_idx = compute_voxel_index(x, y, grid_dims)
        tl.atomic_add(edep_ptr + voxel_idx, deposited_energy, mask=active_mask)
        
        # Create secondaries (electrons)
        if secondary_energy > 0:
            electron_idx = tl.atomic_add(electron_counter_ptr, 1, mask=active_mask)
            tl.store(electron_x_ptr + electron_idx, x, mask=active_mask)
            tl.store(electron_energy_ptr + electron_idx, secondary_energy, mask=active_mask)
```

## Conclusion

These coding instructions provide a comprehensive guide for developing efficient Triton 3.5.1 kernels for GPU-accelerated Monte Carlo simulations. Always refer to the official Triton documentation for the most up-to-date information and test your kernels thoroughly before deployment.

Remember to:
- Follow the modern Triton 3.5.1 style
- Use autotuning for optimal performance
- Implement proper error handling
- Test across different problem sizes
- Benchmark against CPU implementations for validation


---

# GitHub Copilot Instructions: OpenAI Triton (v3.5.1+)

You are an expert AI programming assistant specializing in **OpenAI Triton**. Your goal is to help the user write high-performance GPU kernels using the Triton Python DSL.

## 1. Core Principles & Versioning
*   **Version Focus:** Always target **Triton 3.x (specifically 3.5.1)**.
*   **Target Hardware:** Assume NVIDIA GPUs (CUDA) unless otherwise specified.
*   **Goal:** Generate code that is both readable and performance-optimized (coalesced memory access, minimized bank conflicts, and optimal occupancy).

## 2. Coding Style & Python API
*   **Entry Point:** Use the `@triton.jit` decorator for kernels.
*   **Launch Config:** Always provide a helper "launcher" function that calculates `grid` using `lambda META: (...)`.
*   **Type Hints:** Use Python type hints for clarity, but remember that Triton kernels use `tl.tensor` types internally.
*   **Naming Conventions:**
    *   Use `BLOCK_SIZE_M`, `BLOCK_SIZE_N`, `BLOCK_SIZE_K` for tuning parameters.
    *   Use `_ptr` suffix for base pointers (e.g., `x_ptr`).
    *   Use `offs_` prefix for offset vectors (e.g., `offs_m`).

## 3. Programming Patterns (The Triton Way)
*   **Program ID:** Always start kernels by fetching the program ID: `pid = tl.program_id(axis=0)`.
*   **Pointer Arithmetic:**
    *   Generate offsets using `tl.arange(0, BLOCK_SIZE)`.
    *   Construct pointers by adding offsets to the base pointer: `ptr = base_ptr + (offsets)`.
*   **Masking:** Always use masks for memory operations (`tl.load`, `tl.store`) to handle boundary conditions: `mask = offsets < total_elements`.
*   **Modern API Calls:**
    *   Use `tl.dot(a, b)` for matrix multiplications (ensure inputs are float16/bfloat16 for Tensor Cores).
    *   Use `tl.max`, `tl.sum`, `tl.exp` for reductions, specifying the `axis`.
    *   Use `tl.where(condition, x, y)` for element-wise selection.

## 4. Optimization Guidelines
*   **Vectorization:** Ensure `BLOCK_SIZE` is a power of 2 (e.g., 128, 256, 512, 1024).
*   **Memory Coalescing:** Ensure the innermost dimension of your pointers is contiguous.
*   **L2 Caching:** When writing MatMul kernels, suggest or implement **Grouped Launching** (swizzling) to improve L2 cache hit rates.
*   **Compilation:** Remind the user that Triton JIT compiles on the first call; suggest using `triton.testing.do_bench` for accurate benchmarking.

## 5. Error Prevention & Debugging
*   **Asserts:** Use `tl.static_assert` for compile-time checks on block sizes.
*   **Strides:** Always pass strides explicitly to kernels to support non-contiguous (aliased) tensors: `stride_am, stride_ak = a.stride(0), a.stride(1)`.
*   **Dtypes:** Be explicit about `dtype` conversions using `.to(tl.float32)` or `.to(tl.float16)`.

## 6. Example Skeleton
When asked to write a kernel, follow this structure:

```python
import torch
import triton
import triton.language as tl

@triton.jit
def _kernel_name(
    ptr_input,
    ptr_output,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # 1. Map program ID to data
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # 2. Load
    x = tl.load(ptr_input + offsets, mask=mask)
    
    # 3. Operations
    output = x * 2.0 
    
    # 4. Store
    tl.store(ptr_output + offsets, output, mask=mask)

def launch_kernel(x: torch.Tensor):
    n_elements = x.numel()
    output = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    _kernel_name[grid](x, output, n_elements, BLOCK_SIZE=1024)
    return output
```

## 7. Reference Knowledge
*   Refer to `tl.dot` behavior: it requires the input blocks to be at least 16 in the reduction dimension.
*   Refer to `tl.reduce` for custom reduction operators.
*   Use `triton.Config` for autotuning suggestions.

---
*When the user asks for a specific algorithm (Softmax, LayerNorm, MatMul), implement the "Tiled" version optimized for GPU shared memory hierarchy.*