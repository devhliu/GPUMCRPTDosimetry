# Performance and Benchmarks

This document summarizes the expected performance of the GPUMCRPTDosimetry package and the results of performance benchmarks. The system is designed for high-throughput Monte Carlo simulations on modern NVIDIA GPUs.

## Expected Performance

Performance benchmarks and validation reports indicate the following expected performance on high-end NVIDIA GPUs (e.g., RTX 4090, A4000):

*   **Throughput:** **20-60 seconds per 10⁸ decay histories**. This level of performance makes it feasible to run high-statistic simulations for clinical research and advanced treatment planning.
*   **GPU Memory Usage:** **4-8 GB** for typical simulations (e.g., a 256x256x1024 voxel volume). This is well within the capacity of most modern workstation and data center GPUs.
*   **Kernel Launch Overhead:** **< 0.1 ms** per kernel launch. This is achieved through the extensive use of **CUDA Graphs**, which significantly reduces the overhead associated with launching many small kernels in a loop.
*   **Memory Bandwidth Utilization:** **70-80% of theoretical peak**. This indicates that the GPU kernels, particularly the data layout and memory access patterns, are highly optimized for the GPU architecture.

## Key Performance Optimizations

The high performance of the package is a result of a series of targeted GPU optimizations:

*   **Triton for JIT Compilation:** GPU kernels are written in Triton, allowing for rapid development and JIT compilation into highly efficient machine code.
*   **Wavefront Architecture:** The simulation processes large batches of particles in parallel, which is essential for keeping the GPU's many cores fully utilized.
*   **Structure-of-Arrays (SoA) Data Layout:** This data layout ensures coalesced memory access, which is the single most important factor for achieving high memory bandwidth.
*   **CUDA Graphs with Bucketing:** The use of CUDA Graphs dramatically reduces kernel launch overhead. The "bucketing" strategy allows for graph reuse even with dynamic particle queue sizes.
*   **Lazy Synchronization and GPU-Native Compaction:** The number of synchronization points between the CPU and GPU is minimized. The particle queue compaction process is handled entirely on the GPU using custom Triton kernels.
*   **Philox Counter-Based RNG:** A fast and parallel-friendly RNG that does not require expensive locking or synchronization.

## Test Execution Time

The comprehensive test suite is designed to be fast and efficient, allowing for rapid validation and continuous integration.

| Test Suite                | Number of Tests | Approximate Execution Time (CPU-only) |
|---------------------------|-----------------|---------------------------------------|
| Physics Validation        | 16              | ~1.2 seconds                          |
| Integration Tests         | 13              | ~0.5 seconds                          |
| **Total (CPU-only)**      | **29**          | **~1.7 seconds**                      |
| **GPU Performance Tests** | **8**           | **~5 seconds (variable)**             |

*Note: GPU performance test execution time is dependent on the specific GPU hardware.*

## Bottleneck Analysis

*   **Current Bottleneck:** As is typical for a physics-heavy simulation, the primary bottleneck is **memory bandwidth**. The performance is limited by how fast the GPU can read and write particle data to and from global memory.
*   **Future Optimization Opportunities:**
    *   **Kernel Fusion:** Combining multiple small kernels into a single larger one to reduce register pressure.
    *   **Adaptive Batch Scheduling:** Dynamically adjusting the size of particle batches based on GPU occupancy to further improve utilization.
    *   **Texture Memory:** Using the GPU's texture memory for read-only physics tables could improve cache performance.
