# High-Performance Triton Code: Replacing if Statements

## Summary

Using if statements in GPU kernels can cause warp divergence, where threads in the same warp (32 threads) take different execution paths. This forces the GPU to serialize execution, drastically reducing performance. Triton provides predicate execution mechanisms that execute branch-free code on all threads simultaneously, maintaining full SIMD utilization.

## Key Principles

- Avoid data-dependent branching inside compute loops
- Use `tl.where()` for element-wise conditional logic
- Use `mask=` parameter for conditional memory operations
- Keep if statements only for uniform conditions (e.g., boundary checks)

## Instruction Guide

### Pattern 1: Element-Wise Conditional Logic

#### ❌ Inefficient: Data-dependent if-else

```python
@triton.jit
def inefficient_kernel(x_ptr, y_ptr, output_ptr, n_elements):
    idx = tl.program_id(0)
    if idx < n_elements:
        x = tl.load(x_ptr + idx)
        y = tl.load(y_ptr + idx)
        
        # PROBLEM: Threads diverge based on data values
        if x > 0:
            result = x * y * 2.0
        else:
            result = x + y
        tl.store(output_ptr + idx, result)
```

#### ✅ High-Performance: Use tl.where()

```python
@triton.jit
def efficient_kernel(x_ptr, y_ptr, output_ptr, n_elements):
    idx = tl.program_id(0)
    if idx < n_elements:  # Uniform boundary check is OK
        x = tl.load(x_ptr + idx)
        y = tl.load(y_ptr + idx)
        
        # SOLUTION: Both branches execute, result selected by predicate
        result = tl.where(x > 0, x * y * 2.0, x + y)
        tl.store(output_ptr + idx, result)
```

**Key Benefit:** All threads execute the same instruction sequence, eliminating divergence.

---

### Pattern 2: Conditional Memory Access

#### ❌ Inefficient: if guard for memory operations

```python
# PROBLEM: Breaks memory coalescing, causes divergence
if idx < n_elements:
    data = tl.load(data_ptr + idx)
    processed = data * 2
    tl.store(output_ptr + idx, processed)
```

#### ✅ High-Performance: Use mask= parameter

```python
# SOLUTION: Maintains coalesced memory access
mask = idx < n_elements
data = tl.load(data_ptr + idx, mask=mask)  # Masked load
processed = data * 2
tl.store(output_ptr + idx, processed, mask=mask)  # Masked store
```

**Key Benefit:** Memory accesses remain contiguous and efficient, with predicated execution handled at the hardware level.

---

### Pattern 3: Complex Multi-Branch Logic

#### ❌ Inefficient: Multiple if-elif-else

```python
# PROBLEM: Severe divergence with multiple branches
if x < -1.0:
    result = 0.0
elif x < 0.0:
    result = x + 1.0
elif x < 1.0:
    result = x * x
else:
    result = 1.0
```

#### ✅ High-Performance: Nested tl.where()

```python
# SOLUTION: Flattened predicate execution
result = tl.where(x < -1.0, 0.0,
            tl.where(x < 0.0, x + 1.0,
              tl.where(x < 1.0, x * x, 1.0)))
```

**Key Benefit:** Single execution path with selection happening in parallel across all lanes.

---

### Pattern 4: Early Exit Handling

#### ❌ Inefficient: Conditional return

```python
# PROBLEM: Divergent exit points
if padding_token:
    tl.store(out_ptr + idx, 0.0)
    return
# Continue with computation...
```

#### ✅ High-Performance: Zero-out results with mask

```python
# SOLUTION: Compute anyway, mask final result
mask = ~padding_token  # Invert for valid tokens
result = complex_computation(x, y)
result = tl.where(mask, result, 0.0)  # Zero for padding
tl.store(out_ptr + idx, result)
```

---

## Best Practices Checklist

| Scenario | Recommended Approach | Avoid |
|----------|---------------------|-------|
| Boundary checks | `if idx < n_elements:` (uniform) | Complex conditions |
| Data conditionals | `tl.where(condition, true_val, false_val)` | `if data > 0:` |
| Memory operations | `tl.load(ptr, mask=mask)` | `if valid: tl.load()` |
| Multi-way branches | Nested `tl.where()` | `if-elif-else` chains |
| Early termination | Compute all, mask results | Conditional return |
| Loop handling | Unroll fixed-size loops | `break` inside loops |

---

## Performance Tuning Tips

- **Profile for divergence:** Use NVIDIA Nsight Compute to check "Branch Efficiency" and "Warp Stall Reasons"
- **Minimize register pressure:** `tl.where()` may use extra registers; benchmark different configurations
- **Use autotuner:** Let Triton find optimal `num_warps` for your kernel
- **Structure data layouts:** Ensure condition probabilities are uniform across warps when possible

---

## Conclusion

By consistently applying these patterns, you can achieve 30-50% performance improvements over naive implementations while maintaining readable, maintainable code.
