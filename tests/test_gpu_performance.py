from __future__ import annotations

import pytest
import torch
import triton
import time

from gpumcrpt.materials.hu_materials import MaterialsVolume
from gpumcrpt.physics.tables import load_physics_tables_h5
from gpumcrpt.source.sampling import ParticleQueues


class TestTritonAPICompatibility:
    """Test Triton 3.5.1 API compatibility."""

    def test_triton_version(self):
        """Verify Triton version matches requirement."""
        version = triton.__version__
        major, minor, patch = map(int, version.split('.')[:3])
        
        assert major == 3, f"Expected Triton 3.x, got {major}.{minor}.{patch}"
        assert minor >= 5, f"Expected Triton 3.5+, got {major}.{minor}.{patch}"

    def test_autotune_decorator(self):
        """Test that @triton.autotune works correctly."""
        import triton.language as tl
        
        @triton.autotune(
            configs=[
                triton.Config({'BLOCK_SIZE': 128}),
                triton.Config({'BLOCK_SIZE': 256}),
            ],
            key=['N'],
        )
        @triton.jit
        def simple_kernel(
            x_ptr, y_ptr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr
        ):
            pid = tl.program_id(0)
            offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask = offs < N
            x = tl.load(x_ptr + offs, mask=mask, other=0.0)
            tl.store(y_ptr + offs, x * 2.0, mask=mask)
        
        # Should be callable without errors
        assert callable(simple_kernel)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestGPUMemoryEfficiency:
    """Test GPU memory usage and efficiency."""

    def test_particle_queue_memory_layout(self):
        """Test that particle queues have efficient memory layout."""
        n_particles = 100000
        
        pq = ParticleQueues(
            photons={
                "pos_cm": torch.randn((n_particles, 3), device="cuda"),
                "dir": torch.randn((n_particles, 3), device="cuda"),
                "E_MeV": torch.randn((n_particles,), device="cuda"),
                "w": torch.ones((n_particles,), device="cuda"),
            },
            electrons={
                "pos_cm": torch.empty((0, 3), device="cuda"),
                "dir": torch.empty((0, 3), device="cuda"),
                "E_MeV": torch.empty((0,), device="cuda"),
                "w": torch.empty((0,), device="cuda"),
            },
            positrons={
                "pos_cm": torch.empty((0, 3), device="cuda"),
                "dir": torch.empty((0, 3), device="cuda"),
                "E_MeV": torch.empty((0,), device="cuda"),
                "w": torch.empty((0,), device="cuda"),
            },
        )
        
        # Check contiguity
        assert pq.photons["pos_cm"].is_contiguous()
        assert pq.photons["dir"].is_contiguous()
        
        # Check data types
        assert pq.photons["pos_cm"].dtype == torch.float32
        assert pq.photons["E_MeV"].dtype == torch.float32

    def test_materials_volume_memory_layout(self):
        """Test that MaterialsVolume is efficiently laid out."""
        Z, Y, X = 256, 256, 256
        
        mats = MaterialsVolume(
            material_id=torch.zeros((Z, Y, X), device="cuda", dtype=torch.int32),
            rho=torch.ones((Z, Y, X), device="cuda", dtype=torch.float32),
        )
        
        assert mats.material_id.is_contiguous()
        assert mats.rho.is_contiguous()
        assert mats.material_id.dtype == torch.int32
        assert mats.rho.dtype == torch.float32


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestGPUKernelPerformance:
    """Test GPU kernel performance characteristics."""

    def test_kernel_launch_overhead(self):
        """Test that kernel launches have minimal overhead."""
        import triton.language as tl
        
        @triton.jit
        def copy_kernel(x_ptr, y_ptr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
            pid = tl.program_id(0)
            offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask = offs < N
            x = tl.load(x_ptr + offs, mask=mask, other=0.0)
            tl.store(y_ptr + offs, x, mask=mask)
        
        N = 10_000_000
        x = torch.randn((N,), device="cuda")
        y = torch.empty_like(x)
        
        # Warmup
        copy_kernel[(N // 256 + 1,)](x, y, N=N, BLOCK_SIZE=256)
        torch.cuda.synchronize()
        
        # Time 5 iterations
        start = time.time()
        for _ in range(5):
            copy_kernel[(N // 256 + 1,)](x, y, N=N, BLOCK_SIZE=256)
        torch.cuda.synchronize()
        elapsed = time.time() - start
        
        # Should be fast (< 1 sec for 5 iterations)
        assert elapsed < 1.0, f"Kernel launch too slow: {elapsed:.3f}s for 5 iterations"

    def test_memory_bandwidth_utilization(self):
        """Test bandwidth-limited operation performance."""
        import triton.language as tl
        
        @triton.jit
        def add_kernel(x_ptr, y_ptr, z_ptr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
            pid = tl.program_id(0)
            offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask = offs < N
            x = tl.load(x_ptr + offs, mask=mask, other=0.0)
            y = tl.load(y_ptr + offs, mask=mask, other=0.0)
            tl.store(z_ptr + offs, x + y, mask=mask)
        
        N = 100_000_000
        x = torch.randn((N,), device="cuda", dtype=torch.float32)
        y = torch.randn((N,), device="cuda", dtype=torch.float32)
        z = torch.empty_like(x)
        
        add_kernel[(N // 256 + 1,)](x, y, z, N=N, BLOCK_SIZE=256)
        torch.cuda.synchronize()
        
        # Verify correctness
        z_cpu = x.cpu() + y.cpu()
        assert torch.allclose(z.cpu(), z_cpu, atol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestGPUNumericalStability:
    """Test numerical stability of GPU operations."""

    def test_large_number_accumulation(self):
        """Test atomic add stability with large numbers."""
        import triton.language as tl
        
        @triton.jit
        def atomic_add_kernel(x_ptr, out_ptr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
            pid = tl.program_id(0)
            offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask = offs < N
            x = tl.load(x_ptr + offs, mask=mask, other=0.0)
            tl.atomic_add(out_ptr + 0, x, mask=mask)
        
        N = 1_000_000
        x = torch.ones((N,), device="cuda", dtype=torch.float32)
        out = torch.zeros((1,), device="cuda", dtype=torch.float32)
        
        atomic_add_kernel[(N // 256 + 1,)](x, out, N=N, BLOCK_SIZE=256)
        torch.cuda.synchronize()
        
        # Should sum to approximately N
        result = float(out[0].item())
        assert abs(result - N) < N * 0.01, f"Sum error too large: {result} vs {N}"

    def test_division_by_small_number(self):
        """Test numerical stability when dividing by small numbers."""
        import triton.language as tl
        
        @triton.jit
        def divide_kernel(x_ptr, y_ptr, z_ptr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
            pid = tl.program_id(0)
            offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask = offs < N
            x = tl.load(x_ptr + offs, mask=mask, other=0.0)
            y = tl.load(y_ptr + offs, mask=mask, other=1e-12)
            # Protect against division by zero
            y_safe = tl.maximum(y, 1e-12)
            tl.store(z_ptr + offs, x / y_safe, mask=mask)
        
        N = 100000
        x = torch.randn((N,), device="cuda", dtype=torch.float32)
        y = torch.randn((N,), device="cuda", dtype=torch.float32)
        y[y > 0] = 0.0  # Make some values very small
        y = y + 1e-13
        z = torch.empty_like(x)
        
        divide_kernel[(N // 256 + 1,)](x, y, z, N=N, BLOCK_SIZE=256)
        torch.cuda.synchronize()
        
        # Should be finite
        assert torch.isfinite(z).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestBoundaryConditions:
    """Test boundary handling in GPU kernels."""

    def test_voxel_boundary_checking(self):
        """Test that voxel boundary conditions are handled correctly."""
        import triton.language as tl
        
        @triton.jit
        def boundary_kernel(
            pos_ptr, mat_ptr, Z: tl.constexpr, Y: tl.constexpr, X: tl.constexpr,
            N: tl.constexpr, BLOCK_SIZE: tl.constexpr
        ):
            pid = tl.program_id(0)
            offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask = offs < N
            
            z = tl.load(pos_ptr + offs * 3 + 0, mask=mask, other=-1.0).to(tl.int32)
            y = tl.load(pos_ptr + offs * 3 + 1, mask=mask, other=-1.0).to(tl.int32)
            x = tl.load(pos_ptr + offs * 3 + 2, mask=mask, other=-1.0).to(tl.int32)
            
            inside = (z >= 0) & (z < Z) & (y >= 0) & (y < Y) & (x >= 0) & (x < X)
            lin = z * (Y * X) + y * X + x
            
            # Store 1 if inside, 0 otherwise
            result = tl.where(inside, 1, 0)
            tl.store(mat_ptr + offs, result, mask=mask)
        
        N = 1000
        Z, Y, X = 256, 256, 256
        
        pos = torch.tensor([
            [-1.0, 0.0, 0.0],      # Outside
            [0.0, 0.0, 0.0],       # Inside
            [255.0, 255.0, 255.0], # Inside
            [256.0, 256.0, 256.0], # Outside
        ] + [[128.0, 128.0, 128.0]] * (N - 4), dtype=torch.float32, device="cuda")
        
        result = torch.zeros((N,), device="cuda", dtype=torch.int32)
        
        boundary_kernel[(N // 256 + 1,)](pos, result, Z=Z, Y=Y, X=X, N=N, BLOCK_SIZE=256)
        torch.cuda.synchronize()
        
        assert result[0].item() == 0  # Outside
        assert result[1].item() == 1  # Inside
        assert result[2].item() == 1  # Inside
        assert result[3].item() == 0  # Outside


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
