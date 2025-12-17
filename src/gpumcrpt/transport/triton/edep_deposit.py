from __future__ import annotations

import triton
import triton.language as tl


@triton.jit
def deposit_local_energy_kernel_shared_memory(
    pos_ptr, E_ptr, w_ptr,
    edep_ptr,  # flattened [Z*Y*X]
    Z: tl.constexpr, Y: tl.constexpr, X: tl.constexpr,
    BLOCK: tl.constexpr,
    voxel_z_cm: tl.constexpr, voxel_y_cm: tl.constexpr, voxel_x_cm: tl.constexpr,
    SHARED_MEM_SIZE: tl.constexpr = 1024,  # Size of shared memory buffer
):
    """
    Deposit E*w at the particle position voxel using shared memory to reduce atomic contention.
    Uses a shared memory buffer to accumulate energy deposits locally before writing to global memory.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    
    # Shared memory for local accumulation
    shared_edep = tl.zeros((SHARED_MEM_SIZE,), dtype=tl.float32)
    shared_voxels = tl.zeros((SHARED_MEM_SIZE,), dtype=tl.int32)
    shared_count = tl.zeros((1,), dtype=tl.int32)

    E = tl.load(E_ptr + offs, mask=True, other=0.0)
    w = tl.load(w_ptr + offs, mask=True, other=0.0)

    z = tl.load(pos_ptr + offs * 3 + 0, mask=True, other=0.0)
    y = tl.load(pos_ptr + offs * 3 + 1, mask=True, other=0.0)
    x = tl.load(pos_ptr + offs * 3 + 2, mask=True, other=0.0)

    iz = tl.floor(z / voxel_z_cm).to(tl.int32)
    iy = tl.floor(y / voxel_y_cm).to(tl.int32)
    ix = tl.floor(x / voxel_x_cm).to(tl.int32)
    inside = (iz >= 0) & (iz < Z) & (iy >= 0) & (iy < Y) & (ix >= 0) & (ix < X)
    lin = iz * (Y * X) + iy * X + ix
    
    # Calculate energy deposit
    edep_value = E * w
    
    # Accumulate in shared memory using a simple hashing scheme
    for i in range(BLOCK):
        if inside[i]:
            # Simple hash function to map voxel index to shared memory slot
            voxel_hash = lin[i] % SHARED_MEM_SIZE
            
            # Check if this voxel is already in shared memory
            found = False
            for j in range(SHARED_MEM_SIZE):
                if shared_voxels[j] == lin[i]:
                    # Accumulate to existing entry
                    shared_edep[j] += edep_value[i]
                    found = True
                    break
                elif shared_voxels[j] == 0:  # Empty slot
                    # Add new entry
                    shared_voxels[j] = lin[i]
                    shared_edep[j] = edep_value[i]
                    shared_count[0] += 1
                    found = True
                    break
            
            # If shared memory is full, flush to global memory
            if not found:
                # Flush shared memory to global
                for k in range(SHARED_MEM_SIZE):
                    if shared_voxels[k] != 0:
                        tl.atomic_add(edep_ptr + shared_voxels[k], shared_edep[k])
                        shared_voxels[k] = 0
                        shared_edep[k] = 0.0
                shared_count[0] = 0
                
                # Add current voxel to shared memory
                shared_voxels[0] = lin[i]
                shared_edep[0] = edep_value[i]
                shared_count[0] = 1
    
    # Flush remaining entries in shared memory to global memory
    for k in range(SHARED_MEM_SIZE):
        if shared_voxels[k] != 0:
            tl.atomic_add(edep_ptr + shared_voxels[k], shared_edep[k])


@triton.jit
def deposit_local_energy_kernel(
    pos_ptr, E_ptr, w_ptr,
    edep_ptr,  # flattened [Z*Y*X]
    Z: tl.constexpr, Y: tl.constexpr, X: tl.constexpr,
    BLOCK: tl.constexpr,
    voxel_z_cm: tl.constexpr, voxel_y_cm: tl.constexpr, voxel_x_cm: tl.constexpr,
):
    """
    Deposit E*w at the particle position voxel via atomic add.
    Used for cutoff termination (photon/electron/positron below cutoffs).
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)

    E = tl.load(E_ptr + offs, mask=True, other=0.0)
    w = tl.load(w_ptr + offs, mask=True, other=0.0)

    z = tl.load(pos_ptr + offs * 3 + 0, mask=True, other=0.0)
    y = tl.load(pos_ptr + offs * 3 + 1, mask=True, other=0.0)
    x = tl.load(pos_ptr + offs * 3 + 2, mask=True, other=0.0)

    iz = tl.floor(z / voxel_z_cm).to(tl.int32)
    iy = tl.floor(y / voxel_y_cm).to(tl.int32)
    ix = tl.floor(x / voxel_x_cm).to(tl.int32)
    inside = (iz >= 0) & (iz < Z) & (iy >= 0) & (iy < Y) & (ix >= 0) & (ix < X)
    lin = iz * (Y * X) + iy * X + ix

    tl.atomic_add(edep_ptr + lin, E * w, mask=inside)