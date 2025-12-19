# Performance and Optimization

Achieving high performance is a primary goal of the GPUMCRPTDosimetry project. This document outlines the key optimization strategies employed to accelerate the Monte Carlo simulation on the GPU using Triton.

## Core Optimization Principles

The optimization strategy is built on several fundamental principles of GPU programming:

1.  **Maximize Parallelism:** The wavefront architecture is designed to keep the GPU's many cores busy with a large number of particles being processed in parallel.
2.  **Minimize CPU-GPU Synchronization:** Synchronization between the CPU and GPU is a major performance bottleneck. The design aims to keep as much of the simulation logic as possible on the GPU to minimize these synchronization points.
3.  **Optimize Memory Access:** Efficient use of the GPU's memory hierarchy (global, shared, and register memory) is critical. The primary focus is on ensuring **coalesced memory access** to and from global memory.
4.  **Reduce Kernel Launch Overhead:** The time it takes for the CPU to launch a GPU kernel can be significant, especially if many small kernels are launched in a loop.

## Key Optimization Techniques

### 1. Triton for JIT Compilation
The use of **Triton** allows for the Just-In-Time (JIT) compilation of Python-like code into highly optimized GPU kernels. This provides a significant development speed advantage over traditional CUDA C++ while still offering fine-grained control over performance.

### 2. SoA Data Layout
As described in the `gpu_architecture.md` document, particles are stored in a **Structure-of-Arrays (SoA)** layout. This means that each attribute of the particles (e.g., x-position, y-position, energy) is stored in a separate, contiguous array. When a kernel needs to access the same attribute for many particles, the GPU can read this data in a single, large transaction from global memory, which is known as **coalesced memory access**. This is much more efficient than the alternative Array-of-Structures (AoS) layout.

### 3. Philox Counter-Based RNG
The choice of Random Number Generator (RNG) is critical for both performance and statistical quality in a Monte Carlo simulation. The project uses the **Philox 4x32 counter-based RNG**.
*   **Performance:** Philox is a very fast and lightweight RNG that is well-suited for GPU implementation.
*   **Determinism:** As a counter-based RNG, it allows for deterministic parallel execution. Each particle can be assigned a unique range of random numbers, avoiding the need for expensive locking or synchronization between threads.

### 4. CUDA Graphs
To combat kernel launch overhead, more advanced engines in the project use **CUDA Graphs**.
*   A CUDA Graph captures a sequence of kernel launches and their dependencies.
*   Once captured, the entire sequence can be launched as a single operation, reducing the launch overhead from microseconds to nanoseconds.
*   The project uses **energy-based bucketing** to enable graph reuse. Particles are grouped into buckets based on their energy, and a separate CUDA Graph is captured for each bucket.

### 5. Compaction
After each simulation step, some particles will have "died" (e.g., been absorbed). To maintain performance, these dead particles are removed from the active queues in a process called **compaction**.
*   **Early implementations** used `torch.nonzero` or boolean indexing, which are easy to implement but can have high overhead.
*   **Advanced implementations** use custom Triton kernels that implement a **prefix-sum (scan)** operation to efficiently compact the particle queues directly on the GPU, avoiding CPU-GPU synchronization.

### 6. Lazy Compaction and Single-Sync
The most advanced research prototype (`EM–LazyCompactionSingleSync (Phase 11)`) minimizes CPU-GPU synchronization to an absolute minimum:
*   **Single Sync Point:** The CPU only synchronizes with the GPU once per simulation step to get the number of active particles.
*   **GPU Counter Updates:** All other counter updates and management are handled on the GPU.
*   **Ping-Pong Buffers:** Two sets of particle banks are used ("ping-pong" buffers) to allow kernels to read from one bank and write to the other, avoiding memory hazards without the need for extra synchronization.

These optimizations, taken together, provide a significant performance boost, enabling the simulation of a large number of particle histories in a clinically-acceptable timeframe.
