"""
Performance Monitoring and Profiling Utilities

This module provides utilities for tracking kernel performance, memory usage,
and optimization effectiveness.
"""

from __future__ import annotations

import time
import torch
import triton
from typing import Dict, List, Optional, Any
from dataclasses import dataclass


@dataclass
class KernelProfile:
    """Profile data for a single kernel execution."""
    name: str
    execution_time: float  # seconds
    data_size: int
    block_size: int
    grid_size: int
    memory_usage: int  # bytes
    occupancy: float
    timestamp: float


@dataclass
class MemoryProfile:
    """Memory usage profile."""
    allocated: int
    reserved: int
    peak_allocated: int
    timestamp: float


class PerformanceMonitor:
    """
    Monitors and tracks performance metrics for GPU operations.
    """
    
    def __init__(self, device: torch.device):
        self.device = device
        self.kernel_profiles: List[KernelProfile] = []
        self.memory_profiles: List[MemoryProfile] = []
        self.start_time = time.time()
        
    def record_kernel_execution(self, 
                               name: str,
                               execution_time: float,
                               data_size: int,
                               block_size: int,
                               grid_size: int,
                               memory_usage: int = 0,
                               occupancy: float = 0.0):
        """Record a kernel execution profile."""
        profile = KernelProfile(
            name=name,
            execution_time=execution_time,
            data_size=data_size,
            block_size=block_size,
            grid_size=grid_size,
            memory_usage=memory_usage,
            occupancy=occupancy,
            timestamp=time.time() - self.start_time
        )
        self.kernel_profiles.append(profile)
    
    def record_memory_usage(self):
        """Record current memory usage."""
        if self.device.type == 'cuda':
            allocated = torch.cuda.memory_allocated(self.device)
            reserved = torch.cuda.memory_reserved(self.device)
            
            profile = MemoryProfile(
                allocated=allocated,
                reserved=reserved,
                peak_allocated=torch.cuda.max_memory_allocated(self.device),
                timestamp=time.time() - self.start_time
            )
            self.memory_profiles.append(profile)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get a summary of performance metrics."""
        if not self.kernel_profiles:
            return {}
        
        total_time = sum(p.execution_time for p in self.kernel_profiles)
        total_operations = sum(p.data_size for p in self.kernel_profiles)
        
        # Calculate throughput
        throughput = total_operations / total_time if total_time > 0 else 0
        
        # Find fastest and slowest kernels
        fastest = min(self.kernel_profiles, key=lambda p: p.execution_time)
        slowest = max(self.kernel_profiles, key=lambda p: p.execution_time)
        
        return {
            "total_kernel_executions": len(self.kernel_profiles),
            "total_execution_time": total_time,
            "total_operations": total_operations,
            "average_throughput": throughput,
            "fastest_kernel": fastest.name,
            "fastest_time": fastest.execution_time,
            "slowest_kernel": slowest.name,
            "slowest_time": slowest.execution_time,
            "average_occupancy": sum(p.occupancy for p in self.kernel_profiles) / len(self.kernel_profiles),
        }
    
    def print_performance_report(self):
        """Print a formatted performance report."""
        summary = self.get_performance_summary()
        
        if not summary:
            print("No performance data recorded.")
            return
        
        print("\n" + "="*60)
        print("GPU PERFORMANCE REPORT")
        print("="*60)
        print(f"Total kernel executions: {summary['total_kernel_executions']}")
        print(f"Total execution time: {summary['total_execution_time']:.3f} seconds")
        print(f"Total operations: {summary['total_operations']:,}")
        print(f"Average throughput: {summary['average_throughput']:,.0f} ops/sec")
        print(f"Average occupancy: {summary['average_occupancy']:.2%}")
        print(f"Fastest kernel: {summary['fastest_kernel']} ({summary['fastest_time']:.6f}s)")
        print(f"Slowest kernel: {summary['slowest_kernel']} ({summary['slowest_time']:.6f}s)")
        
        # Memory usage if available
        if self.memory_profiles:
            latest_mem = self.memory_profiles[-1]
            print(f"Current memory usage: {latest_mem.allocated / 1024**2:.1f} MB")
            print(f"Peak memory usage: {latest_mem.peak_allocated / 1024**2:.1f} MB")
        
        print("="*60)


def time_kernel_execution(func):
    """
    Decorator to time kernel execution and record performance metrics.
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        execution_time = time.time() - start_time
        
        # Extract performance information if available
        data_size = kwargs.get('data_size', 0)
        block_size = kwargs.get('block_size', 256)
        grid_size = kwargs.get('grid_size', 1)
        
        # Record performance if monitor is provided
        monitor = kwargs.get('performance_monitor')
        if monitor and isinstance(monitor, PerformanceMonitor):
            monitor.record_kernel_execution(
                name=func.__name__,
                execution_time=execution_time,
                data_size=data_size,
                block_size=block_size,
                grid_size=grid_size
            )
        
        return result
    
    return wrapper


def profile_memory_usage(func):
    """
    Decorator to profile memory usage during function execution.
    """
    def wrapper(*args, **kwargs):
        device = kwargs.get('device', torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        
        if device.type == 'cuda':
            # Reset peak memory stats
            torch.cuda.reset_peak_memory_stats(device)
            
            # Record initial memory usage
            initial_allocated = torch.cuda.memory_allocated(device)
            initial_reserved = torch.cuda.memory_reserved(device)
            
            result = func(*args, **kwargs)
            
            # Record final memory usage
            final_allocated = torch.cuda.memory_allocated(device)
            final_reserved = torch.cuda.memory_reserved(device)
            peak_allocated = torch.cuda.max_memory_allocated(device)
            
            print(f"\nMemory Usage for {func.__name__}:")
            print(f"  Initial allocated: {initial_allocated / 1024**2:.1f} MB")
            print(f"  Final allocated: {final_allocated / 1024**2:.1f} MB")
            print(f"  Peak allocated: {peak_allocated / 1024**2:.1f} MB")
            print(f"  Memory delta: {(final_allocated - initial_allocated) / 1024**2:.1f} MB")
            
            return result
        else:
            return func(*args, **kwargs)
    
    return wrapper


def analyze_memory_access_pattern(tensor: torch.Tensor, 
                                access_indices: Optional[torch.Tensor] = None) -> Dict[str, Any]:
    """
    Analyze memory access patterns for optimization insights.
    
    Args:
        tensor: Tensor to analyze
        access_indices: Optional indices representing access pattern
        
    Returns:
        Dictionary with analysis results
    """
    analysis = {}
    
    # Basic tensor properties
    analysis["shape"] = tensor.shape
    analysis["stride"] = tensor.stride()
    analysis["is_contiguous"] = tensor.is_contiguous()
    analysis["memory_layout"] = "contiguous" if tensor.is_contiguous() else "non-contiguous"
    
    # Memory alignment analysis
    element_size = tensor.element_size()
    base_address = tensor.data_ptr()
    analysis["alignment"] = base_address % 128  # 128-byte alignment for coalescing
    
    # Access pattern analysis if indices provided
    if access_indices is not None:
        # Calculate stride patterns
        if access_indices.numel() > 1:
            strides = torch.diff(access_indices)
            analysis["access_stride_mean"] = strides.float().mean().item()
            analysis["access_stride_std"] = strides.float().std().item()
            analysis["coalescing_efficiency"] = calculate_coalescing_efficiency(strides)
    
    return analysis


def calculate_coalescing_efficiency(strides: torch.Tensor) -> float:
    """
    Calculate memory coalescing efficiency (0.0 to 1.0).
    
    Args:
        strides: Tensor of memory access strides
        
    Returns:
        Coalescing efficiency (higher is better)
    """
    if strides.numel() == 0:
        return 1.0
    
    # Ideal case: sequential access (stride = 1)
    ideal_strides = torch.ones_like(strides, dtype=torch.float32)
    
    # Calculate deviation from ideal
    deviation = torch.abs(strides.float() - ideal_strides)
    efficiency = 1.0 - (deviation.mean() / (ideal_strides.max() - ideal_strides.min()))
    
    return max(0.0, min(1.0, efficiency.item()))