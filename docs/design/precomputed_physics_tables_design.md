# Precomputed Physics Tables Design Document

## Executive Summary

This document presents a comprehensive design for improving the pre-computed physics tables system in the GPUMCRPTDosimetry package. The current system uses placeholder physics data and lacks comprehensive support for different simulation modes and materials. The proposed design implements real physics calculations based on established databases, provides enhanced materials management, and optimizes for GPU performance.

## Current System Analysis

### Materials System (`src/gpumcrpt/materials/`)
- **Materials Registry**: Manages multiple HU-to-material tables with YAML configuration
- **HU Materials**: Defines material libraries and volume classes
- **Materials Manager**: Handles materials data and precomputation
- **Current Limitations**: Limited material definitions, basic HU mapping

### Physics System (`src/gpumcrpt/physics/`)
- **Physics Tables**: Generates HDF5 tables for different simulation methods
- **Relaxation Tables**: Handles atomic relaxation processes
- **Current Limitations**: Placeholder physics data, incomplete physics processes

### Simulation Modes
1. **local_deposit**: Simplified energy deposition
2. **photon_only**: Photon transport only
3. **photon_em_condensed**: Photon transport with condensed history electron transport

## Proposed Architecture

### Enhanced Materials System

#### 1. Comprehensive Material Definitions

```python
@dataclass
class EnhancedMaterial:
    name: str
    category: str  # "air", "soft_tissue", "bone", "custom"
    density_g_cm3: float
    elemental_composition: List[ElementalComposition]
    effective_Z: float
    electron_density_relative_to_water: float
    source: str  # "ICRU-44", "NIST", "Custom"
```

#### 2. Improved HU-to-Material Mapping
- Piecewise linear interpolation for density conversion
- Material classification based on standard HU ranges
- Validation of material assignments

### Physics Calculation Framework

#### 1. Base Calculator Architecture
```python
class BasePhysicsCalculator(ABC):
    def __init__(self, parameters: PhysicsParameters):
        self.parameters = parameters
        self.energy_grid = self._create_energy_grid()
    
    @abstractmethod
    def calculate_for_material(self, material, energy: torch.Tensor) -> torch.Tensor:
        pass
```

#### 2. Physics Processes Implementation

**Photon Interactions:**
- **Photoelectric Effect**: Empirical formulas with edge corrections
- **Compton Scattering**: Klein-Nishina cross-sections
- **Rayleigh Scattering**: Form factor approximations
- **Pair Production**: Bethe-Heitler cross-sections

**Electron Interactions:**
- **Stopping Power**: Bethe-Bloch formula with density corrections
- **Bremsstrahlung**: Simplified yield calculations
- **Multiple Scattering**: Gaussian approximation

### Table Storage Strategy

#### HDF5 Schema Version 2.0
```
/physics_tables/
├── metadata/
│   ├── version
│   ├── creation_date
│   ├── method
│   └── validation_info
├── energy/
│   ├── centers_MeV
│   └── edges_MeV
├── materials/
│   ├── names
│   ├── densities
│   ├── effective_Z
│   └── electron_densities
├── photon_interactions/
│   ├── photoelectric
│   ├── compton
│   ├── rayleigh
│   ├── pair_production
│   └── total
├── electron_interactions/
│   ├── stopping_power_restricted
│   ├── stopping_power_unrestricted
│   ├── csda_range
│   └── brems_yield
└── sampling_tables/
    ├── compton_inv_cdf
    ├── rayleigh_inv_cdf
    └── brems_inv_cdf
```

### Simulation Mode-Specific Tables

#### 1. local_deposit Mode
- **Purpose**: Fast dose calculation for treatment planning
- **Required Tables**: Photon attenuation coefficients, energy deposition kernels
- **Optimization**: Coarse energy grid, minimal sampling tables

#### 2. photon_only Mode
- **Purpose**: Photon transport studies
- **Required Tables**: Complete photon interaction cross-sections, sampling tables
- **Optimization**: Fine energy grid, comprehensive sampling

#### 3. photon_em_condensed Mode
- **Purpose**: Full photon-electron transport
- **Required Tables**: All photon and electron interactions, condensed history parameters
- **Optimization**: Balanced accuracy and performance

## Implementation Plan

### Phase 1: Foundation (Weeks 1-2)
1. **Enhanced Materials System**
   - Implement comprehensive material definitions
   - Create enhanced HU-to-material mapping
   - Add material validation framework

2. **Physics Calculator Framework**
   - Design base calculator architecture
   - Implement energy grid creation utilities
   - Create parameter validation system

### Phase 2: Core Physics (Weeks 3-6)
1. **Photon Interaction Calculations**
   - Photoelectric cross-sections
   - Compton scattering (Klein-Nishina)
   - Rayleigh scattering
   - Pair production

2. **Electron Interaction Calculations**
   - Stopping powers (Bethe-Bloch)
   - Bremsstrahlung yields
   - Multiple scattering parameters

### Phase 3: Table Generation (Weeks 7-8)
1. **Comprehensive Table Generator**
   - Method-specific table generation
   - HDF5 schema implementation
   - Compression and optimization

2. **Sampling Table Generation**
   - Inverse CDF creation
   - Angular distribution sampling
   - Energy loss sampling

### Phase 4: Integration (Weeks 9-10)
1. **Transport Engine Updates**
   - Modify existing engines to use new tables
   - Performance optimization
   - Validation testing

2. **Validation Framework**
   - Comparison with NIST databases
   - Accuracy assessment
   - Performance benchmarking

## Key Design Decisions

### 1. Physics Accuracy vs Performance
- **Decision**: Prioritize physics accuracy for clinical applications
- **Rationale**: Medical dosimetry requires high accuracy for patient safety
- **Compromise**: Use optimized algorithms and GPU acceleration

### 2. Storage Format
- **Decision**: HDF5 with compression
- **Rationale**: Standard format, efficient storage, cross-platform compatibility
- **Alternative Considered**: Custom binary format (rejected for maintainability)

### 3. Material Representation
- **Decision**: Elemental composition with effective properties
- **Rationale**: Accurate physics calculations, compatibility with established databases
- **Alternative Considered**: Simple density-only approach (insufficient for accurate physics)

### 4. Energy Grid Design
- **Decision**: Logarithmic spacing with method-specific resolution
- **Rationale**: Better coverage of low-energy physics, optimized for each simulation mode
- **Alternative Considered**: Uniform spacing (inefficient for wide energy ranges)

## Performance Considerations

### Memory Optimization
- **GPU Memory**: Use PyTorch tensors with appropriate data types
- **Table Compression**: HDF5 gzip compression for storage efficiency
- **Lazy Loading**: Load tables on-demand with caching

### Computational Efficiency
- **Vectorized Operations**: Leverage PyTorch for GPU acceleration
- **Precomputation**: Generate tables offline to avoid runtime calculations
- **Optimized Sampling**: Use efficient inverse CDF methods

### Storage Strategy
- **Directory Structure**: Organized by simulation method and physics process
- **Version Control**: Schema versioning for backward compatibility
- **Validation Data**: Include checksums and validation information

## Validation Framework

### 1. Physics Validation
- **NIST XCOM**: Compare cross-sections with established database
- **ICRU Reports**: Validate against published data
- **Monte Carlo Codes**: Cross-verify with EGSnrc, Geant4

### 2. Clinical Validation
- **Phantom Studies**: Compare with measured data
- **Treatment Planning**: Validate against clinical systems
- **Accuracy Metrics**: Dose difference, gamma analysis

### 3. Performance Validation
- **Speed Tests**: Compare with current implementation
- **Memory Usage**: Monitor GPU and system memory
- **Scalability**: Test with different problem sizes

## File Structure

```
src/gpumcrpt/
├── materials/
│   ├── enhanced_materials.py          # Enhanced material definitions
│   ├── hu_mapping_enhanced.py         # Improved HU mapping
│   ├── materials_registry.py          # Registry management
│   └── materials_manager.py           # Enhanced manager
├── physics/
│   ├── calculators/                   # Physics calculation framework
│   │   ├── base_calculator.py
│   │   ├── photoelectric_calculator.py
│   │   ├── compton_calculator.py
│   │   ├── rayleigh_calculator.py
│   │   ├── pair_production_calculator.py
│   │   ├── stopping_power_calculator.py
│   │   └── bremsstrahlung_calculator.py
│   ├── tables/                        # Table generation system
│   │   ├── table_generator.py
│   │   ├── sampling_generator.py
│   │   └── table_manager.py
│   └── validation/                    # Validation framework
│       ├── nist_validator.py
│       ├── accuracy_metrics.py
│       └── performance_benchmark.py
└── transport/                         # Updated transport engines
    ├── engine_gpu_triton_local_deposit.py
    ├── engine_gpu_triton_photon_only.py
    └── engine_gpu_triton_photon_em_condensed.py

data/
├── materials/                         # Material configuration files
│   ├── standard_tissues.yaml
│   ├── human_organs.yaml
│   └── custom_materials.yaml
└── physics_tables/                    # Generated physics tables
    ├── local_deposit_physics.h5
    ├── photon_only_physics.h5
    └── photon_em_condensed_physics.h5
```

## API Design

### Table Generation API
```python
class PhysicsTableGenerator:
    def generate_tables(
        self,
        method: str,
        output_path: str,
        config: TableGenerationConfig
    ) -> None:
        """Generate physics tables for specified method."""

class TableGenerationConfig:
    method: str
    energy_range_MeV: Tuple[float, float]
    num_energy_points: int
    include_sampling_tables: bool
    validation_enabled: bool
```

### Table Management API
```python
class PhysicsTableManager:
    def load_tables(
        self,
        method: str,
        device: str = "cuda"
    ) -> Dict[str, torch.Tensor]:
        """Load physics tables for simulation."""
    
    def get_table_info(self, method: str) -> TableInfo:
        """Get metadata about available tables."""
```

## Expected Benefits

### 1. Physics Accuracy
- **Current**: Placeholder data with limited accuracy
- **Improved**: Real physics calculations based on established databases
- **Impact**: Clinically relevant dose calculations

### 2. Performance
- **Current**: Basic implementation with potential bottlenecks
- **Improved**: GPU-optimized data structures and algorithms
- **Impact**: Faster simulations for clinical workflow

### 3. Maintainability
- **Current**: Hard-coded physics data
- **Improved**: Modular, extensible architecture
- **Impact**: Easier updates and enhancements

### 4. Clinical Relevance
- **Current**: Limited material definitions
- **Improved**: Comprehensive medical material library
- **Impact**: Better representation of human tissues

## Risk Assessment

### Technical Risks
1. **Physics Accuracy**: Mitigated by validation against established databases
2. **Performance**: Addressed through GPU optimization and efficient algorithms
3. **Compatibility**: Maintained through careful API design and versioning

### Implementation Risks
1. **Schedule**: Phased approach with clear milestones
2. **Complexity**: Modular design with focused components
3. **Testing**: Comprehensive validation framework

## Conclusion

This design provides a comprehensive roadmap for transforming the GPUMCRPTDosimetry physics tables system from a placeholder implementation to a clinically accurate, high-performance solution. The modular architecture ensures maintainability and extensibility, while the focus on real physics calculations and validation ensures clinical relevance and accuracy.

The implementation will proceed in four phases, with each phase delivering tangible improvements. The final system will support all three simulation modes with appropriate optimizations for each use case, providing a solid foundation for clinical dosimetry applications.