# Photodiode DEVSIM Demo

## Purpose

Interactive demo for exploring photodiode device physics by experimenting with different materials, doping concentrations, and material parameters. This tool lets you quickly prototype and compare photodiode design.

## What You Can Explore

**Material Properties**
- Permittivity, intrinsic carrier density, electron charge
- Carrier lifetimes (taun, taup)
- Absorption coefficients for different wavelengths

**Device Parameters**
- P-region and N-region doping levels
- Carrier mobility models (Caughey-Thomas parameters)
- Device dimensions and geometry

**Performance Analysis**
- I-V characteristics (dark vs illuminated)
- External quantum efficiency at different wavelengths
- Current density and collection efficiency

## Quick Start

```python
# Create simulator
simulator = PhotodiodeSimulator(
    device_name="demo_photodiode",
    mesh_file="output/photodiode_mesh.msh"
)

# Try different materials/doping
simulator.p_doping = 1e15  # cm^-3
simulator.n_doping = 1e18  # cm^-3
simulator.wavelength_nm = 850  # Change wavelength

# Run simulation
simulator.run_complete_simulation()
```

## Key Features

- **Material parameter playground** - Easy modification of silicon properties
- **Interactive plots** - Real-time visualization of I-V curves and quantum efficiency  
- **Robust solver** - Two-step equilibrium with adaptive voltage stepping
- **Diagnostic output** - Detailed feedback for understanding device behavior

## Future Integration

This demo framework is designed to connect with atomistic simulations, allowing material parameters derived from first-principles calculations to feed directly into the device-level simulation.

## File Structure

- `main.py` - Main simulation class and workflow
- `physic_builder.py` - Core semiconductor physics models  
- `compute_iv.py` - I-V sweep analysis
- `compute_qe.py` - Quantum efficiency calculations
- `set_material.py` - Material parameter configuration
- `mobility_model.py` - Carrier mobility modeling
- `solve_equilibrium.py` - Initial condition solver

## Requirements

- DEVSIM device simulator
- NumPy for numerical operations
- Plotly for interactive visualization
- GMSH mesh file (basic 2D p-n junction geometry)

## Note on Accuracy

Should be improved.