from __future__ import annotations

import torch
from typing import Dict, Tuple


class GPUConfigOptimizer:
    """
    Optimizes kernel configurations based on GPU architecture and workload.
    """
    
    def __init__(self, device: torch.device):
        self.device = device
        self.arch_info = self._get_gpu_architecture_info()
        self.cache_sizes = self._get_cache_sizes()
        
    def _get_gpu_architecture_info(self) -> Dict:
        """Get GPU architecture information."""
        if self.device.type != 'cuda':
            return {"arch": "unknown", "compute_capability": (0, 0)}
            
        try:
            props = torch.cuda.get_device_properties(self.device)
            return {
                "arch": props.name,
                "compute_capability": props.major + props.minor / 10,
                "sm_count": props.multi_processor_count,
                "max_threads_per_block": props.max_threads_per_block,
                "warp_size": props.warp_size,
                "shared_memory_per_block": props.shared_memory_per_block,
                "registers_per_block": props.registers_per_block,
            }
        except:
            return {"arch": "unknown", "compute_capability": (0, 0)}
    
    def _get_cache_sizes(self) -> Dict:
        """Get approximate cache sizes for different GPU architectures."""
        cc = self.arch_info.get("compute_capability", (0, 0))
        
        cc_float = cc[0] + cc[1] / 10.0
        
        if cc_float >= 8.0:
            return {"l1": 128 * 1024, "l2": 40 * 1024 * 1024}
        elif cc_float >= 7.0:
            return {"l1": 96 * 1024, "l2": 6 * 1024 * 1024}
        elif cc_float >= 6.0:
            return {"l1": 48 * 1024, "l2": 4 * 1024 * 1024}
        else:
            return {"l1": 48 * 1024, "l2": 1.5 * 1024 * 1024}
    
    def optimize_block_size(self, 
                           data_size: int, 
                           shared_memory_required: int = 0,
                           register_pressure: int = 32) -> int:
        """
        Optimize block size for given workload.
        
        Args:
            data_size: Number of elements to process
            shared_memory_required: Shared memory required per block (bytes)
            register_pressure: Estimated register usage per thread
            
        Returns:
            Optimal block size
        """
        max_threads = self.arch_info.get("max_threads_per_block", 1024)
        warp_size = self.arch_info.get("warp_size", 32)
        shared_mem_per_block = self.arch_info.get("shared_memory_per_block", 48 * 1024)
        regs_per_block = self.arch_info.get("registers_per_block", 65536)
        
        candidate_sizes = [32, 64, 128, 256, 512, 1024]
        candidate_sizes = [s for s in candidate_sizes if s <= max_threads]
        
        if shared_memory_required > 0:
            max_threads_by_shared = shared_mem_per_block // max(1, shared_memory_required)
            candidate_sizes = [s for s in candidate_sizes if s <= max_threads_by_shared]
        
        if register_pressure > 0:
            max_threads_by_regs = regs_per_block // register_pressure
            candidate_sizes = [s for s in candidate_sizes if s <= max_threads_by_regs]
        
        if not candidate_sizes:
            return min(256, max_threads)
        
        optimal_size = candidate_sizes[-1]
        
        if data_size < optimal_size:
            for size in candidate_sizes:
                if size >= data_size:
                    optimal_size = size
                    break
        
        return optimal_size
    
    def optimize_grid_size(self, data_size: int, block_size: int) -> int:
        """
        Optimize grid size for given data size and block size.
        
        Args:
            data_size: Number of elements to process
            block_size: Block size to use
            
        Returns:
            Optimal grid size
        """
        sm_count = self.arch_info.get("sm_count", 80)
        
        min_grid = (data_size + block_size - 1) // block_size
        
        target_grid = sm_count * 3
        
        return max(min_grid, min(target_grid, 65535))


def optimize_memory_access_pattern(data: torch.Tensor, 
                                  access_pattern: str = "coalesced") -> torch.Tensor:
    """
    Adjust memory access patterns for GPU efficiency.
    
    Args:
        data: Input tensor
        access_pattern: Desired access pattern ("coalesced", "bank_conflict_free")
        
    Returns:
        Adjusted tensor
    """
    if access_pattern == "coalesced":
        if not data.is_contiguous():
            data = data.contiguous()
        
        if data.numel() > 0:
            element_size = data.element_size()
            optimal_alignment = 128 // element_size
            if data.numel() % optimal_alignment != 0:
                pass
    
    return data


def create_soa_layout(data_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Create Structure-of-Arrays (SoA) layout for GPU access.
    
    Args:
        data_dict: Dictionary of tensors to organize
        
    Returns:
        Dictionary with SoA layout
    """
    result_dict = {}
    
    for key, tensor in data_dict.items():
        result_tensor = tensor.contiguous()
        
        if tensor.dtype == torch.float64:
            result_tensor = result_tensor.to(torch.float32)
        elif tensor.dtype == torch.int64:
            result_tensor = result_tensor.to(torch.int32)
        
        result_dict[key] = result_tensor
    
    return result_dict


def estimate_occupancy(block_size: int, 
                      shared_mem_per_block: int,
                      registers_per_thread: int) -> float:
    """
    Estimate GPU occupancy for given kernel configuration.
    
    Args:
        block_size: Threads per block
        shared_mem_per_block: Shared memory usage per block
        registers_per_thread: Register usage per thread
        
    Returns:
        Estimated occupancy (0.0 to 1.0)
    """
    max_threads_per_sm = 2048
    max_blocks_per_sm = 32
    max_shared_mem_per_sm = 96 * 1024
    max_registers_per_sm = 65536
    
    threads_limit = max_threads_per_sm // block_size
    shared_mem_limit = max_shared_mem_per_sm // max(1, shared_mem_per_block)
    registers_limit = max_registers_per_sm // (block_size * max(1, registers_per_thread))
    
    max_blocks = min(threads_limit, shared_mem_limit, registers_limit, max_blocks_per_sm)
    
    return min(1.0, max_blocks * block_size / max_threads_per_sm)


def get_optimal_kernel_config(data_size: int,
                            device: torch.device,
                            shared_mem_required: int = 0,
                            register_pressure: int = 32) -> Tuple[int, int]:
    """
    Get optimal kernel configuration for given workload.
    
    Args:
        data_size: Number of elements to process
        device: Target device
        shared_mem_required: Shared memory required per block
        register_pressure: Register usage per thread
        
    Returns:
        Tuple of (block_size, grid_size)
    """
    optimizer = GPUConfigOptimizer(device)
    
    block_size = optimizer.optimize_block_size(
        data_size, shared_mem_required, register_pressure
    )
    
    grid_size = optimizer.optimize_grid_size(data_size, block_size)
    
    return block_size, grid_size
