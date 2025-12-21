# Physics Tables Implementation Plan

## Overview

This document provides a detailed implementation plan for the improved pre-computed physics tables system. It includes specific code examples, file structures, and step-by-step implementation guidelines.

## Phase 1: Enhanced Materials System (Weeks 1-2)

### 1.1 Expand Materials Library

**File**: `src/gpumcrpt/materials/enhanced_materials.py`

```python
from dataclasses import dataclass
from typing import Dict, List, Optional
import torch

@dataclass
class ElementalComposition:
    """Detailed elemental composition for materials."""
    symbol: str
    atomic_number: int
    mass_fraction: float
    atomic_fraction: Optional[float] = None

@dataclass  
class EnhancedMaterial:
    """Enhanced material definition with comprehensive properties."""
    name: str
    category: str  # "air", "soft_tissue", "bone", "custom"
    density_g_cm3: float
    elemental_composition: List[ElementalComposition]
    effective_Z: float
    electron_density_relative_to_water: float
    source: str  # "ICRU-44", "NIST", "Custom"
    
    def to_tensor_format(self) -> Dict[str, torch.Tensor]:
        """Convert to tensor format for GPU computation."""
        symbols = [elem.symbol for elem in self.elemental_composition]
        Z_values = [elem.atomic_number for elem in self.elemental_composition]
        mass_fractions = [elem.mass_fraction for elem in self.elemental_composition]
        
        return {
            'symbols': symbols,
            'Z': torch.tensor(Z_values, dtype=torch.int32),
            'mass_fractions': torch.tensor(mass_fractions, dtype=torch.float32)
        }

class EnhancedMaterialsLibrary:
    """Comprehensive materials library with validation."""
    
    def __init__(self):
        self.materials: Dict[str, EnhancedMaterial] = {}
        self._load_standard_materials()
    
    def _load_standard_materials(self):
        """Load standard medical physics materials."""
        # ICRU-44 standard tissues
        self.materials['air'] = EnhancedMaterial(
            name='air', category='air', density_g_cm3=0.0012,
            elemental_composition=[
                ElementalComposition('N', 7, 0.755),
                ElementalComposition('O', 8, 0.232),
                ElementalComposition('Ar', 18, 0.013)
            ],
            effective_Z=7.64, electron_density_relative_to_water=0.001,
            source='ICRU-44'
        )
        
        # Add more standard materials...
    
    def validate_composition(self, material: EnhancedMaterial) -> bool:
        """Validate material composition sums to 1.0."""
        total_mass = sum(elem.mass_fraction for elem in material.elemental_composition)
        return abs(total_mass - 1.0) < 1e-6
```

### 1.2 Enhanced HU-to-Material Mapping

**File**: `src/gpumcrpt/materials/hu_mapping_enhanced.py`

```python
import numpy as np
import torch
from typing import List, Tuple

class EnhancedHUMapping:
    """Enhanced HU to material mapping with validation."""
    
    def __init__(self, materials_library: EnhancedMaterialsLibrary):
        self.library = materials_library
        self.hu_ranges = self._define_standard_hu_ranges()
    
    def _define_standard_hu_ranges(self) -> List[Tuple[float, float, str]]:
        """Define standard HU ranges for medical imaging."""
        return [
            (-1000, -850, 'air'),      # Air
            (-850, -910, 'lung'),      # Lung tissue
            (-100, -50, 'fat'),        # Fat
            (0, 50, 'soft_tissue'),    # Soft tissue
            (40, 100, 'muscle'),       # Muscle
            (300, 800, 'trabecular_bone'),  # Trabecular bone
            (800, 2000, 'cortical_bone')   # Cortical bone
        ]
    
    def hu_to_material(self, hu_values: torch.Tensor) -> torch.Tensor:
        """Convert HU values to material indices."""
        material_indices = torch.zeros_like(hu_values, dtype=torch.int32)
        
        for i, (hu_min, hu_max, material_name) in enumerate(self.hu_ranges):
            mask = (hu_values >= hu_min) & (hu_values < hu_max)
            material_indices[mask] = i
        
        return material_indices
    
    def hu_to_density(self, hu_values: torch.Tensor) -> torch.Tensor:
        """Convert HU values to density using piecewise linear interpolation."""
        # Define HU-density calibration points
        hu_points = [-1000, -850, -100, 0, 40, 300, 800, 2000]
        density_points = [0.0012, 0.26, 0.95, 1.00, 1.06, 1.16, 1.85, 2.00]
        
        return torch.interp(hu_values, 
                          torch.tensor(hu_points, dtype=torch.float32),
                          torch.tensor(density_points, dtype=torch.float32))
```

## Phase 2: Physics Calculation Framework (Weeks 3-6)

### 2.1 Base Physics Calculator

**File**: `src/gpumcrpt/physics/calculators/base_calculator.py`

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
import torch

@dataclass
class PhysicsParameters:
    """Parameters for physics calculations."""
    energy_range_MeV: Tuple[float, float] = (0.01, 20.0)
    num_energy_points: int = 200
    log_spacing: bool = True
    interpolation_method: str = 'linear'
    
class BasePhysicsCalculator(ABC):
    """Base class for physics calculations."""
    
    def __init__(self, parameters: PhysicsParameters):
        self.parameters = parameters
        self.energy_grid = self._create_energy_grid()
    
    def _create_energy_grid(self) -> torch.Tensor:
        """Create energy grid based on parameters."""
        if self.parameters.log_spacing:
            energies = np.logspace(
                np.log10(self.parameters.energy_range_MeV[0]),
                np.log10(self.parameters.energy_range_MeV[1]),
                self.parameters.num_energy_points
            )
        else:
            energies = np.linspace(
                self.parameters.energy_range_MeV[0],
                self.parameters.energy_range_MeV[1],
                self.parameters.num_energy_points
            )
        
        return torch.tensor(energies, dtype=torch.float32)
    
    @abstractmethod
    def calculate_for_material(self, material, energy: torch.Tensor) -> torch.Tensor:
        """Calculate physics quantity for a specific material."""
        pass
    
    def calculate_for_all_materials(self, materials: List) -> torch.Tensor:
        """Calculate physics quantity for all materials."""
        results = []
        for material in materials:
            result = self.calculate_for_material(material, self.energy_grid)
            results.append(result)
        
        return torch.stack(results)  # Shape: [num_materials, num_energies]
```

### 2.2 Photoelectric Cross-Section Calculator

**File**: `src/gpumcrpt/physics/calculators/photoelectric_calculator.py`

```python
import numpy as np
import torch
from .base_calculator import BasePhysicsCalculator, PhysicsParameters

class PhotoelectricCalculator(BasePhysicsCalculator):
    """Calculate photoelectric cross-sections using empirical formulas."""
    
    def calculate_for_material(self, material, energy: torch.Tensor) -> torch.Tensor:
        """Calculate photoelectric cross-section for a material."""
        # Empirical formula: σ_pe ∝ Z^4 / E^3
        # More accurate implementation would use EPDL97 data
        
        Z_eff = material.effective_Z
        energy_keV = energy * 1000  # Convert to keV
        
        # Base cross-section at 100 keV for water
        sigma_base = 0.03  # cm²/g at 100 keV for water (Z=7.42)
        
        # Scale by Z^4 and energy^-3
        Z_water = 7.42
        scaling = (Z_eff / Z_water) ** 4
        energy_scaling = (100.0 / energy_keV) ** 3
        
        cross_sections = sigma_base * scaling * energy_scaling
        
        # Apply corrections for low energies
        # Add edge effects and other corrections here
        
        return cross_sections
```

### 2.3 Compton Scattering Calculator

**File**: `src/gpumcrpt/physics/calculators/compton_calculator.py`

```python
import numpy as np
import torch
from .base_calculator import BasePhysicsCalculator

class ComptonCalculator(BasePhysicsCalculator):
    """Calculate Compton scattering cross-sections using Klein-Nishina formula."""
    
    def calculate_for_material(self, material, energy: torch.Tensor) -> torch.Tensor:
        """Calculate Compton cross-section per electron, then scale by electron density."""
        
        # Klein-Nishina cross-section per electron
        electron_cross_sections = self._klein_nishina(energy)
        
        # Scale by electron density of material
        electron_density = material.electron_density_relative_to_water
        # Water has ~3.34e23 electrons/cm³
        water_electron_density = 3.34e23
        
        cross_sections = electron_cross_sections * electron_density * water_electron_density
        
        return cross_sections
    
    def _klein_nishina(self, energy: torch.Tensor) -> torch.Tensor:
        """Calculate Klein-Nishina cross-section per electron."""
        # Electron rest mass in MeV
        m_e = 0.511  # MeV
        
        # Classical electron radius in cm
        r_e = 2.818e-13  # cm
        
        alpha = energy / m_e
        
        # Klein-Nishina formula
        term1 = (1 + alpha) / (alpha ** 2)
        term2 = (2 * (1 + alpha) / (1 + 2 * alpha) - 
                torch.log(1 + 2 * alpha) / alpha)
        term3 = torch.log(1 + 2 * alpha) / (2 * alpha)
        term4 = (1 + 3 * alpha) / ((1 + 2 * alpha) ** 2)
        
        sigma = (np.pi * r_e ** 2) * (term1 * (term2 + term3 + term4))
        
        return sigma
```

### 2.4 Electron Stopping Power Calculator

**File**: `src/gpumcrpt/physics/calculators/stopping_power_calculator.py`

```python
import numpy as np
import torch
from .base_calculator import BasePhysicsCalculator

class StoppingPowerCalculator(BasePhysicsCalculator):
    """Calculate electron stopping powers using Bethe-Bloch formula."""
    
    def calculate_for_material(self, material, energy: torch.Tensor) -> torch.Tensor:
        """Calculate restricted stopping power for electrons."""
        
        # Simplified Bethe-Bloch formula for electrons
        # dE/dx = (2πN_A r_e^2 m_e c^2 Z) / (A β^2) * [ln(...) - δ/2]
        
        Z_eff = material.effective_Z
        density = material.density_g_cm3
        
        # Constants
        N_A = 6.022e23  # Avogadro's number
        r_e = 2.818e-13  # cm
        m_e_c2 = 0.511  # MeV
        
        # Electron velocity (β = v/c)
        gamma = 1 + energy / m_e_c2
        beta = torch.sqrt(1 - 1 / (gamma ** 2))
        
        # Mean excitation energy (simplified)
        I = 10 * Z_eff * 1e-6  # MeV
        
        # Bethe-Bloch formula (simplified)
        K = (2 * np.pi * N_A * r_e ** 2 * m_e_c2) / density
        ln_term = torch.log(2 * m_e_c2 * beta ** 2 * gamma ** 2 / I)
        
        stopping_power = (K * Z_eff / (beta ** 2)) * ln_term
        
        # Apply density effect correction (simplified)
        delta = self._density_effect_correction(energy, density, Z_eff)
        stopping_power -= delta / 2
        
        return torch.clamp(stopping_power, min=0.1, max=100.0)
    
    def _density_effect_correction(self, energy, density, Z_eff):
        """Simplified density effect correction."""
        # Placeholder implementation
        return 0.1 * torch.log(energy + 0.1)
```

## Phase 3: Table Generation System (Weeks 7-8)

### 3.1 Comprehensive Table Generator

**File**: `src/gpumcrpt/physics/tables/table_generator.py`

```python
from dataclasses import dataclass
from typing import Dict, List, Optional
import h5py
import numpy as np
import torch

from ..calculators import (
    PhotoelectricCalculator, ComptonCalculator, 
    StoppingPowerCalculator, PhysicsParameters
)
from ...materials.enhanced_materials import EnhancedMaterialsLibrary

@dataclass
class TableGenerationConfig:
    """Configuration for table generation."""
    method: str  # 'local_deposit', 'photon_only', 'photon_em_condensed'
    energy_range_MeV: Tuple[float, float] = (0.01, 20.0)
    num_energy_points: int = 200
    include_sampling_tables: bool = True
    validation_enabled: bool = True
    compression: bool = True

class PhysicsTableGenerator:
    """Generate comprehensive physics tables for different simulation methods."""
    
    def __init__(self, config: TableGenerationConfig):
        self.config = config
        self.materials_lib = EnhancedMaterialsLibrary()
        self.calculators = self._initialize_calculators()
    
    def _initialize_calculators(self) -> Dict[str, BasePhysicsCalculator]:
        """Initialize physics calculators based on configuration."""
        params = PhysicsParameters(
            energy_range_MeV=self.config.energy_range_MeV,
            num_energy_points=self.config.num_energy_points
        )
        
        calculators = {
            'photoelectric': PhotoelectricCalculator(params),
            'compton': ComptonCalculator(params),
            'stopping_power': StoppingPowerCalculator(params)
        }
        
        return calculators
    
    def generate_tables(self, output_path: str) -> None:
        """Generate comprehensive physics tables."""
        
        with h5py.File(output_path, 'w') as f:
            # Metadata
            self._write_metadata(f)
            
            # Materials information
            self._write_materials_info(f)
            
            # Energy grid
            self._write_energy_grid(f)
            
            # Physics tables based on method
            if self.config.method == 'local_deposit':
                self._generate_local_deposit_tables(f)
            elif self.config.method == 'photon_only':
                self._generate_photon_only_tables(f)
            elif self.config.method == 'photon_em_condensed':
                self._generate_photon_em_condensed_tables(f)
            
            # Sampling tables if enabled
            if self.config.include_sampling_tables:
                self._generate_sampling_tables(f)
            
            # Validation data if enabled
            if self.config.validation_enabled:
                self._write_validation_info(f)
    
    def _generate_photon_em_condensed_tables(self, f: h5py.File) -> None:
        """Generate tables for photon-electron condensed history method."""
        
        # Photon interactions
        photon_group = f.create_group('photon_interactions')
        
        materials = list(self.materials_lib.materials.values())
        
        # Photoelectric cross-sections
        photo_xs = self.calculators['photoelectric'].calculate_for_all_materials(materials)
        photon_group.create_dataset('photoelectric', data=photo_xs.numpy(), 
                                  compression='gzip' if self.config.compression else None)
        
        # Compton cross-sections
        compton_xs = self.calculators['compton'].calculate_for_all_materials(materials)
        photon_group.create_dataset('compton', data=compton_xs.numpy(),
                                  compression='gzip' if self.config.compression else None)
        
        # Electron interactions
        electron_group = f.create_group('electron_interactions')
        
        # Stopping powers
        stopping_powers = self.calculators['stopping_power'].calculate_for_all_materials(materials)
        electron_group.create_dataset('stopping_power_restricted', 
                                    data=stopping_powers.numpy(),
                                    compression='gzip' if self.config.compression else None)
```

### 3.2 Sampling Table Generator

**File**: `src/gpumcrpt/physics/tables/sampling_generator.py`

```python
import numpy as np
import torch
from typing import Tuple

class SamplingTableGenerator:
    """Generate inverse CDF tables for sampling distributions."""
    
    def generate_compton_inv_cdf(self, energies: torch.Tensor, 
                               num_samples: int = 1000) -> torch.Tensor:
        """Generate inverse CDF for Compton scattering angular distribution."""
        
        # Klein-Nishina differential cross-section
        # dσ/dΩ ∝ [1 + cos²θ + (E/mc²)²(1-cosθ)² / (1 + (E/mc²)(1-cosθ))]
        
        num_energies = len(energies)
        cos_theta = torch.linspace(-1, 1, num_samples)
        
        inv_cdf = torch.zeros((num_energies, num_samples))
        
        for i, energy in enumerate(energies):
            # Calculate PDF for this energy
            pdf = self._klein_nishina_pdf(energy, cos_theta)
            
            # Normalize PDF
            pdf = pdf / pdf.sum()
            
            # Calculate CDF
            cdf = torch.cumsum(pdf, dim=0)
            cdf = cdf / cdf[-1]  # Normalize to 1.0
            
            # Create inverse CDF
            inv_cdf[i] = self._create_inverse_cdf(cdf, cos_theta)
        
        return inv_cdf
    
    def _klein_nishina_pdf(self, energy: float, cos_theta: torch.Tensor) -> torch.Tensor:
        """Calculate Klein-Nishina PDF for given energy and cos(theta)."""
        m_e = 0.511  # MeV
        alpha = energy / m_e
        
        term1 = 1 + cos_theta ** 2
        term2 = (alpha ** 2) * (1 - cos_theta) ** 2
        term3 = 1 + alpha * (1 - cos_theta)
        
        pdf = term1 + term2 / term3
        
        return pdf
    
    def _create_inverse_cdf(self, cdf: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        """Create inverse CDF mapping from uniform [0,1] to distribution values."""
        num_samples = 1000
        uniform_samples = torch.linspace(0, 1, num_samples)
        
        inv_cdf = torch.zeros(num_samples)
        
        for i, u in enumerate(uniform_samples):
            # Find where CDF first exceeds u
            idx = torch.searchsorted(cdf, u)
            idx = torch.clamp(idx, 0, len(values) - 1)
            inv_cdf[i] = values[idx]
        
        return inv_cdf
```

## Phase 4: Integration and Optimization (Weeks 9-10)

### 4.1 Enhanced Table Manager

**File**: `src/gpumcrpt/physics/tables/table_manager.py`

```python
from dataclasses import dataclass
from typing import Dict, Optional
import os
import h5py
import torch

@dataclass
class TableCache:
    """Cache for frequently accessed physics tables."""
    tables: Dict[str, torch.Tensor]
    access_count: Dict[str, int]
    max_cache_size: int = 10

class PhysicsTableManager:
    """Manage loading and caching of physics tables."""
    
    def __init__(self, tables_directory: str = "data/physics_tables"):
        self.tables_directory = tables_directory
        self.cache = TableCache(tables={}, access_count={})
        self._ensure_directory()
    
    def load_tables(self, method: str, device: str = "cuda") -> Dict[str, torch.Tensor]:
        """Load physics tables for specified method."""
        
        cache_key = f"{method}_{device}"
        
        # Check cache first
        if cache_key in self.cache.tables:
            self.cache.access_count[cache_key] += 1
            return self.cache.tables[cache_key]
        
        # Load from file
        table_path = os.path.join(self.tables_directory, f"{method}_physics.h5")
        
        if not os.path.exists(table_path):
            raise FileNotFoundError(f"Physics tables not found: {table_path}")
        
        tables = self._load_h5_tables(table_path, device)
        
        # Update cache
        self._update_cache(cache_key, tables)
        
        return tables
    
    def _load_h5_tables(self, file_path: str, device: str) -> Dict[str, torch.Tensor]:
        """Load tables from HDF5 file."""
        tables = {}
        
        with h5py.File(file_path, 'r') as f:
            # Load energy grid
            if 'energy/centers_MeV' in f:
                tables['energy_centers'] = torch.tensor(
                    f['energy/centers_MeV'][:], device=device, dtype=torch.float32
                )
            
            # Load photon interactions
            if 'photon_interactions' in f:
                photon_group = f['photon_interactions']
                for key in photon_group.keys():
                    tables[f'photon_{key}'] = torch.tensor(
                        photon_group[key][:], device=device, dtype=torch.float32
                    )
            
            # Load electron interactions
            if 'electron_interactions' in f:
                electron_group = f['electron_interactions']
                for key in electron_group.keys():
                    tables[f'electron_{key}'] = torch.tensor(
                        electron_group[key][:], device=device, dtype=torch.float32
                    )
        
        return tables
    
    def _update_cache(self, key: str, tables: Dict[str, torch.Tensor]) -> None:
        """Update cache with new tables."""
        # If cache is full, remove least recently used
        if len(self.cache.tables) >= self.cache.max_cache_size:
            lru_key = min(self.cache.access_count.items(), key=lambda x: x[1])[0]
            del self.cache.tables[lru_key]
            del self.cache.access_count[lru_key]
        
        self.cache.tables[key] = tables
        self.cache.access_count[key] = 1
```

## File Structure Summary

```
src/gpumcrpt/
├── materials/
│   ├── enhanced_materials.py      # Enhanced material definitions
│   ├── hu_mapping_enhanced.py     # Improved HU-to-material mapping
│   └── materials_registry.py      # Registry for multiple material configurations
├── physics/
│   ├── calculators/
│   │   ├── base_calculator.py     # Base physics calculator class
│   │   ├── photoelectric_calculator.py
│   │   ├── compton_calculator.py
│   │   ├── rayleigh_calculator.py
│   │   ├── pair_production_calculator.py
│   │   └── stopping_power_calculator.py
│   ├── tables/
│   │   ├── table_generator.py     # Main table generation
│   │   ├── sampling_generator.py  # Inverse CDF generation
│   │   └── table_manager.py       # Table loading and caching
│   └── validation/
│       ├── nist_validator.py      # Validation against NIST data
│       └── accuracy_metrics.py    # Accuracy assessment
└── transport/
    └── engine_gpu_triton_*.py     # Updated to use new tables
```

## Implementation Timeline

### Week 1-2: Materials System Enhancement
- [ ] Implement EnhancedMaterialsLibrary
- [ ] Create comprehensive material definitions
- [ ] Implement enhanced HU mapping
- [ ] Add validation for material compositions

### Week 3-4: Core Physics Calculators
- [ ] Implement base calculator framework
- [ ] Develop photoelectric cross-section calculator
- [ ] Implement Compton scattering calculator
- [ ] Create electron stopping power calculator

### Week 5-6: Advanced Physics
- [ ] Implement Rayleigh scattering calculator
- [ ] Develop pair production calculator
- [ ] Add bremsstrahlung and delta-ray calculations
- [ ] Create sampling table generators

### Week 7-8: Table Generation System
- [ ] Implement comprehensive table generator
- [ ] Add HDF5 schema version 2.0
- [ ] Create sampling table generation
- [ ] Implement table validation framework

### Week 9: Integration
- [ ] Update transport engines to use new tables
- [ ] Implement table caching system
- [ ] Add performance optimizations
- [ ] Create unit tests

### Week 10: Validation and Documentation
- [ ] Validate against NIST databases
- [ ] Performance benchmarking
- [ ] Comprehensive documentation
- [ ] Example usage scripts

## Testing Strategy

### Unit Tests
- Material composition validation
- Physics calculation accuracy
- Table generation correctness
- HDF5 file integrity

### Integration Tests
- End-to-end table generation pipeline
- Transport engine compatibility
- Performance regression testing

### Validation Tests
- Comparison with NIST XCOM data
- Verification against established Monte Carlo codes
- Clinical accuracy assessment

This implementation plan provides a comprehensive roadmap for developing a robust, physics-accurate pre-computed tables system that will significantly improve the accuracy and performance of the GPUMCRPTDosimetry package.