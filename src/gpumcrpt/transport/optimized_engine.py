"""
Optimized GPU Engine with RNG Enhancement and Memory Access Improvements

This module provides an optimized version of the GPU transport engine that integrates:
1. Philox counter-based RNG system
2. Memory access pattern optimizations
3. Dynamic kernel configuration
4. Performance monitoring
"""

from __future__ import annotations

import time
import torch
import triton
from typing import Dict, Any, Optional

from gpumcrpt.transport.triton.optimization import GPUConfigOptimizer, create_soa_layout, optimize_memory_access_pattern
from gpumcrpt.transport.triton.performance import PerformanceMonitor, time_kernel_execution
from gpumcrpt.kernels.triton.rng_bridge import upgrade_rng_i32_to_philox_soa, is_philox_soa
from gpumcrpt.transport.triton.photon_flight_optimized import launch_optimized_photon_flight


class OptimizedGPUEngine:
    """
    Optimized GPU transport engine with enhanced RNG and memory access patterns.
    """
    
    def __init__(self, sim_config: Dict[str, Any], device: torch.device):
        self.sim_config = sim_config
        self.device = device
        self.performance_monitor = PerformanceMonitor(device)
        self.config_optimizer = GPUConfigOptimizer(device)
        
        # Track optimization state
        self.optimization_enabled = True
        self.use_philox_rng = True
        self.use_dynamic_config = True
        
    def initialize_particle_bank(self, n_particles: int) -> Dict[str, torch.Tensor]:
        """
        Initialize particle bank with optimized SoA layout and Philox RNG.
        """
        bank = {}
        
        # Geometry (SoA layout)
        bank['x'] = torch.zeros(n_particles, dtype=torch.float32, device=self.device)
        bank['y'] = torch.zeros(n_particles, dtype=torch.float32, device=self.device)
        bank['z'] = torch.zeros(n_particles, dtype=torch.float32, device=self.device)
        bank['dx'] = torch.zeros(n_particles, dtype=torch.float32, device=self.device)
        bank['dy'] = torch.zeros(n_particles, dtype=torch.float32, device=self.device)
        bank['dz'] = torch.zeros(n_particles, dtype=torch.float32, device=self.device)
        
        # Physics
        bank['E'] = torch.zeros(n_particles, dtype=torch.float32, device=self.device)
        bank['w'] = torch.zeros(n_particles, dtype=torch.float32, device=self.device)
        bank['ebin'] = torch.zeros(n_particles, dtype=torch.int32, device=self.device)
        
        # Philox RNG state
        if self.use_philox_rng:
            seed = int(self.sim_config.get("seed", 1234))
            bank['rng_key0'] = torch.full((n_particles,), seed & 0xFFFFFFFF, 
                                        device=self.device, dtype=torch.int32)
            # Use proper integer arithmetic to avoid overflow
            key1_value = (seed * 2654435761) & 0xFFFFFFFF
            # Ensure the value fits in int32 range
            if key1_value > 0x7FFFFFFF:
                key1_value = key1_value - 0x100000000  # Convert to signed int32
            bank['rng_key1'] = torch.full((n_particles,), key1_value,
                                        device=self.device, dtype=torch.int32)
            bank['rng_ctr0'] = torch.arange(n_particles, device=self.device, dtype=torch.int32)
            bank['rng_ctr1'] = torch.zeros((n_particles,), device=self.device, dtype=torch.int32)
            bank['rng_ctr2'] = torch.zeros((n_particles,), device=self.device, dtype=torch.int32)
            bank['rng_ctr3'] = torch.zeros((n_particles,), device=self.device, dtype=torch.int32)
        else:
            # Legacy RNG for compatibility
            g = torch.Generator(device=self.device)
            g.manual_seed(int(self.sim_config.get("seed", 0)))
            bank['rng'] = torch.randint(1, 2**31 - 1, (n_particles,), 
                                      generator=g, device=self.device, dtype=torch.int32)
        
        # Optimize memory layout
        if self.optimization_enabled:
            bank = create_soa_layout(bank)
            
        return bank
    
    def optimize_memory_layout(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optimize memory layout for GPU access patterns.
        """
        if not self.optimization_enabled:
            return data_dict
        
        optimized_dict = {}
        
        for key, value in data_dict.items():
            # Only optimize tensor values, leave other types unchanged
            if isinstance(value, torch.Tensor):
                # Ensure contiguous memory
                optimized_tensor = optimize_memory_access_pattern(value, "coalesced")
                
                # Convert to optimal dtypes
                if value.dtype == torch.float64:
                    optimized_tensor = optimized_tensor.to(torch.float32)
                elif value.dtype == torch.int64:
                    optimized_tensor = optimized_tensor.to(torch.int32)
                
                optimized_dict[key] = optimized_tensor
            else:
                # Leave non-tensor values unchanged
                optimized_dict[key] = value
        
        return optimized_dict
    
    @time_kernel_execution
    def run_optimized_photon_transport(self, 
                                     photon_bank: Dict[str, torch.Tensor],
                                     geometry_data: Dict[str, Any],
                                     physics_tables: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Run optimized photon transport with enhanced RNG and memory access.
        """
        # Ensure Philox RNG is used
        if self.use_philox_rng and not is_philox_soa(photon_bank):
            photon_bank = upgrade_rng_i32_to_philox_soa(photon_bank)
        
        # Optimize memory layout
        photon_bank = self.optimize_memory_layout(photon_bank)
        geometry_data = self.optimize_memory_layout(geometry_data)
        physics_tables = self.optimize_memory_layout(physics_tables)
        
        # Prepare output bank
        n_particles = len(photon_bank['E'])
        output_bank = self.initialize_particle_bank(n_particles)
        
        # Extract required data
        pos = torch.stack([photon_bank['x'], photon_bank['y'], photon_bank['z']], dim=1)
        direction = torch.stack([photon_bank['dx'], photon_bank['dy'], photon_bank['dz']], dim=1)
        
        out_pos = torch.stack([output_bank['x'], output_bank['y'], output_bank['z']], dim=1)
        out_direction = torch.stack([output_bank['dx'], output_bank['dy'], output_bank['dz']], dim=1)
        
        # Launch optimized photon flight
        if self.use_philox_rng:
            launch_optimized_photon_flight(
                # Inputs
                pos=pos,
                direction=direction,
                E=photon_bank['E'],
                w=photon_bank['w'],
                rng_key0=photon_bank['rng_key0'],
                rng_key1=photon_bank['rng_key1'],
                rng_ctr0=photon_bank['rng_ctr0'],
                rng_ctr1=photon_bank['rng_ctr1'],
                rng_ctr2=photon_bank['rng_ctr2'],
                rng_ctr3=photon_bank['rng_ctr3'],
                ebin=photon_bank['ebin'],
                
                # Outputs
                out_pos=out_pos,
                out_direction=out_direction,
                out_E=output_bank['E'],
                out_w=output_bank['w'],
                out_rng_key0=output_bank['rng_key0'],
                out_rng_key1=output_bank['rng_key1'],
                out_rng_ctr0=output_bank['rng_ctr0'],
                out_rng_ctr1=output_bank['rng_ctr1'],
                out_rng_ctr2=output_bank['rng_ctr2'],
                out_rng_ctr3=output_bank['rng_ctr3'],
                out_ebin=output_bank['ebin'],
                out_alive=torch.empty(n_particles, dtype=torch.int8, device=self.device),
                out_real=torch.empty(n_particles, dtype=torch.int8, device=self.device),
                
                # Geometry and materials
                material_id=geometry_data['material_id'],
                rho=geometry_data['rho'],
                sigma_total=physics_tables['sigma_total'],
                sigma_max=physics_tables['sigma_max'],
                ref_rho=physics_tables['ref_rho'],
                
                # Simulation parameters
                voxel_size_cm=geometry_data['voxel_size_cm'],
                device=self.device,
                
                # Performance monitoring
                performance_monitor=self.performance_monitor if self.optimization_enabled else None
            )
        else:
            # Fallback to legacy implementation
            # (Implementation would go here)
            pass
        
        # Update output bank with position/direction data
        output_bank['x'] = out_pos[:, 0]
        output_bank['y'] = out_pos[:, 1]
        output_bank['z'] = out_pos[:, 2]
        output_bank['dx'] = out_direction[:, 0]
        output_bank['dy'] = out_direction[:, 1]
        output_bank['dz'] = out_direction[:, 2]
        
        return output_bank
    
    def enable_optimizations(self, enable: bool = True):
        """Enable or disable optimizations."""
        self.optimization_enabled = enable
        self.use_philox_rng = enable
        self.use_dynamic_config = enable
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get performance report with optimization metrics."""
        return self.performance_monitor.get_performance_summary()
    
    def print_optimization_status(self):
        """Print current optimization status."""
        print("\n" + "="*50)
        print("GPU OPTIMIZATION STATUS")
        print("="*50)
        print(f"Optimizations enabled: {self.optimization_enabled}")
        print(f"Philox RNG: {self.use_philox_rng}")
        print(f"Dynamic kernel config: {self.use_dynamic_config}")
        
        # GPU architecture info
        arch_info = self.config_optimizer.arch_info
        print(f"GPU Architecture: {arch_info.get('arch', 'Unknown')}")
        print(f"Compute Capability: {arch_info.get('compute_capability', 'Unknown')}")
        print(f"SM Count: {arch_info.get('sm_count', 'Unknown')}")
        
        # Performance summary
        if self.performance_monitor.kernel_profiles:
            summary = self.performance_monitor.get_performance_summary()
            print(f"\nPerformance Summary:")
            print(f"  Total kernel executions: {summary['total_kernel_executions']}")
            print(f"  Total execution time: {summary['total_execution_time']:.3f}s")
            print(f"  Average throughput: {summary['average_throughput']:,.0f} ops/sec")
        
        print("="*50)


def benchmark_optimizations(base_engine, optimized_engine, test_data: Dict[str, Any]):
    """
    Benchmark performance improvements from optimizations.
    
    Args:
        base_engine: Engine without optimizations
        optimized_engine: Engine with optimizations
        test_data: Test data for benchmarking
    """
    print("\n" + "="*60)
    print("OPTIMIZATION BENCHMARK")
    print("="*60)
    
    # Warm-up runs
    for _ in range(3):
        _ = base_engine.run_optimized_photon_transport(**test_data)
        _ = optimized_engine.run_optimized_photon_transport(**test_data)
    
    # Benchmark runs
    n_runs = 10
    base_times = []
    optimized_times = []
    
    for i in range(n_runs):
        # Base engine
        start_time = time.time()
        result_base = base_engine.run_optimized_photon_transport(**test_data)
        base_times.append(time.time() - start_time)
        
        # Optimized engine
        start_time = time.time()
        result_optimized = optimized_engine.run_optimized_photon_transport(**test_data)
        optimized_times.append(time.time() - start_time)
        
        print(f"Run {i+1}/{n_runs}: Base={base_times[-1]:.3f}s, Optimized={optimized_times[-1]:.3f}s")
    
    # Calculate statistics
    base_avg = sum(base_times) / n_runs
    optimized_avg = sum(optimized_times) / n_runs
    speedup = base_avg / optimized_avg
    
    print("\nBenchmark Results:")
    print(f"  Base engine average: {base_avg:.3f}s")
    print(f"  Optimized engine average: {optimized_avg:.3f}s")
    print(f"  Speedup: {speedup:.2f}x")
    print(f"  Performance improvement: {(speedup - 1) * 100:.1f}%")
    
    # Print optimization report
    optimized_engine.print_optimization_status()
    
    print("="*60)
    
    return {
        'base_avg_time': base_avg,
        'optimized_avg_time': optimized_avg,
        'speedup': speedup,
        'performance_improvement': (speedup - 1) * 100
    }