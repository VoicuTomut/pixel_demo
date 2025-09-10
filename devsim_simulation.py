#!/usr/bin/env python3
"""
Photodiode Device Simulation using DEVSIM

This module simulates a silicon photodiode device, calculating I-V characteristics,
quantum efficiency, and other key parameters under various operating conditions.

"""

import numpy as np
import plotly.graph_objects as go


import devsim

# Import custom modules
from pixi.compute_qe import calculate_qe
from pixi.doping_profile import define_doping
from pixi.mobility_model import define_mobility_models
from pixi.set_material import set_material_parameters
from pixi.physic_builder import build_physic_model
from pixi.boundary import set_boundary
from pixi.compute_iv import run_iv_sweep
from pixi.solve_equilibrium import solve_equilibrium
from pixi import initialize_device_and_mesh, debug_device_and_mesh, print_debug_summary
from material_parameters import silicon_material_properties


class PhotodiodeSimulator:
    """
    A comprehensive photodiode device simulator using DEVSIM.

    This class encapsulates all the functionality needed to simulate
    a silicon photodiode, including mesh setup, physics modeling,
    and characterization measurements.
    """

    def __init__(self, device_name: str = "photodiode", mesh_file: str = "output/photodiode_mesh.msh"):
        """Initialize the photodiode simulator."""
        self.device_name = device_name
        self.mesh_file = mesh_file

        # Simulation parameters
        self.photon_flux = 0.0  # photons/cm²/s (0.0 for dark simulation)
        self.absorption_coefficient = 1e4  # 1/cm
        self.material_descriptor = silicon_material_properties

        # Doping concentrations
        self.p_doping = 1e15  # cm⁻³
        self.n_doping = 1e18  # cm⁻³

        # Device geometry
        self.device_width_cm = 4.0e-4  # cm
        self.wavelength_nm = 650  # nm

        # Analysis parameters
        self.light_photon_flux = 1e17  # photons/cm²/s
        self.thermal_voltage = 0.0259  # V

        # Storage for results
        self.dark_currents = None
        self.light_currents = None
        self.qe_values = None
        self.iv_voltages = None

    def initialize_device(self) -> None:
        """Initialize device mesh and verify setup."""
        print("=" * 80)
        print("INITIALIZING PHOTODIODE DEVICE")
        print("=" * 80)

        print(f"Device name: {self.device_name}")
        print(f"Mesh file: {self.mesh_file}")

        # Initialize device and mesh
        initialize_device_and_mesh(self.device_name, self.mesh_file)

        # Debug and verify mesh
        debug_info = debug_device_and_mesh(self.device_name, self.mesh_file)
        print_debug_summary(debug_info)

    def setup_materials(self) -> None:
        """Configure material parameters for both regions."""
        print("\n" + "-" * 60)
        print("SETTING UP MATERIAL PARAMETERS")
        print("-" * 60)

        regions = ["p_region", "n_region"]

        for region in regions:
            print(f"  Setting silicon parameters for {region}...")
            set_material_parameters(
                device=self.device_name,
                region=region,
                material_descriptor=self.material_descriptor
            )

    def setup_doping_and_mobility(self) -> None:
        """Configure doping profiles and mobility models."""
        print("\n" + "-" * 60)
        print("SETTING UP DOPING AND MOBILITY")
        print("-" * 60)

        print(f"  P-region doping: {self.p_doping:.1e} cm⁻³")
        print(f"  N-region doping: {self.n_doping:.1e} cm⁻³")

        define_doping(
            device=self.device_name,
            p_doping=self.p_doping,
            n_doping=self.n_doping
        )

        print("  Defining mobility models...")
        for region in ["p_region", "n_region"]:
            define_mobility_models(device=self.device_name, region=region)

    def setup_physics_model(self) -> None:
        """Set up the complete physics model for the photodiode."""
        print("\n" + "=" * 80)
        print("SETTING UP PHYSICS MODEL")
        print("=" * 80)

        # Create solution variables
        print("  Creating solution variables (Potential, Electrons, Holes)...")
        self._create_solution_variables()

        # Build physics model
        print("  Building bulk physics models...")
        build_physic_model(
            self.device_name,
            self.photon_flux,
            self.absorption_coefficient,
            thermal_voltage=self.thermal_voltage
        )

        # Set boundary conditions
        print("  Setting boundary conditions...")
        set_boundary(self.device_name)

        # Solve for equilibrium
        print("  Solving for initial equilibrium...")
        solve_equilibrium(self.device_name)

        print("✅ Physics model setup complete!")

    def _create_solution_variables(self) -> None:
        """Create solution variables for both regions."""
        variables = ["Potential", "Electrons", "Holes"]
        regions = ["p_region", "n_region"]

        for region in regions:
            for variable in variables:
                devsim.node_solution(
                    device=self.device_name,
                    region=region,
                    name=variable
                )

    def _generate_voltage_array(self) -> np.ndarray:
        """Generate adaptive voltage stepping for better convergence."""
        # Fine steps near zero for better resolution
        initial_steps = np.linspace(0.0, -0.5, 30)

        # Medium steps for moderate reverse bias
        medium_steps = np.linspace(-0.6, -2.0, 15)

        # Larger steps for high reverse bias
        large_steps = np.linspace(-2.2, -5.0, 12)

        # Combine and sort in reverse order (0V to -5V)
        voltages = np.unique(np.concatenate([initial_steps, medium_steps, large_steps]))
        return voltages[::-1]

    def run_dark_current_simulation(self) -> np.ndarray:
        """Run I-V sweep in dark conditions."""
        print("\n" + "-" * 60)
        print("RUNNING DARK CURRENT SIMULATION")
        print("-" * 60)

        self.iv_voltages = self._generate_voltage_array()
        print(f"  Voltage range: {self.iv_voltages[0]:.1f}V to {self.iv_voltages[-1]:.1f}V")
        print(f"  Number of points: {len(self.iv_voltages)}")

        self.dark_currents = run_iv_sweep(
            self.device_name,
            self.iv_voltages,
            p_flux=0.0
        )

        print("✅ Dark current simulation complete!")
        return self.dark_currents

    def run_photocurrent_simulation(self) -> np.ndarray:
        """Run I-V sweep under illumination."""
        print("\n" + "-" * 60)
        print("RUNNING PHOTOCURRENT SIMULATION")
        print("-" * 60)

        # Stabilize at reverse bias before ramping photon flux
        print("  Stabilizing at reverse bias...")
        devsim.set_parameter(
            device=self.device_name,
            name="anode_bias",
            value=self.iv_voltages[-1]
        )
        devsim.solve(type="dc", absolute_error=1e10, relative_error=10)

        # Gradually ramp up photon flux for stability
        print("  Ramping photon flux for stability...")
        self._ramp_photon_flux()

        # Run illuminated I-V sweep
        print(f"  Running I-V sweep with flux = {self.light_photon_flux:.1e}")
        self.light_currents = run_iv_sweep(
            self.device_name,
            self.iv_voltages,
            p_flux=self.light_photon_flux
        )

        print("✅ Photocurrent simulation complete!")
        return self.light_currents

    def _ramp_photon_flux(self) -> None:
        """Gradually increase photon flux for numerical stability."""
        flux_ramp = np.logspace(12, np.log10(self.light_photon_flux), 6)

        for i, flux in enumerate(flux_ramp):
            devsim.set_parameter(name="PhotonFlux", value=flux)
            print(f"    Step {i + 1}/{len(flux_ramp)}: "
                  f"PhotonFlux = {flux:.1e} @ V = {self.iv_voltages[-1]:.1f}V")
            devsim.solve(
                type="dc",
                absolute_error=1e10,
                relative_error=10,
                maximum_iterations=100
            )

    def calculate_quantum_efficiency(self) -> np.ndarray:
        """Calculate external quantum efficiency."""
        print("\n" + "-" * 60)
        print("CALCULATING QUANTUM EFFICIENCY")
        print("-" * 60)

        if self.dark_currents is None or self.light_currents is None:
            raise ValueError("Must run both dark and light I-V sweeps first!")

        self.qe_values, _ = calculate_qe(
            self.dark_currents,
            self.light_currents,
            self.light_photon_flux,
            self.device_width_cm,
            self.wavelength_nm
        )

        print(f"  Wavelength: {self.wavelength_nm} nm")
        print(f"  Device width: {self.device_width_cm} cm")
        print(f"  Max QE: {np.max(self.qe_values):.2f}%")
        print("✅ Quantum efficiency calculation complete!")

        return self.qe_values

    def print_simulation_summary(self) -> None:
        """Print detailed summary of simulation results."""
        print("\n" + "=" * 80)
        print("SIMULATION SUMMARY")
        print("=" * 80)

        if self.dark_currents is not None and self.light_currents is not None:
            max_current_diff = max(
                self.dark_currents - self.light_currents,
                key=abs
            )

            print(f"Dark current range: {np.min(np.abs(self.dark_currents)):.2e} "
                  f"to {np.max(np.abs(self.dark_currents)):.2e} A/cm")
            print(f"Photocurrent range: {np.min(np.abs(self.light_currents)):.2e} "
                  f"to {np.max(np.abs(self.light_currents)):.2e} A/cm")
            print(f"Maximum current difference: {max_current_diff:.2e} A/cm")

            if self.qe_values is not None:
                print(f"QE range: {np.min(self.qe_values):.2f}% to {np.max(self.qe_values):.2f}%")

    def create_interactive_plots(self) -> None:
        """Generate interactive plots using Plotly."""
        print("\n" + "-" * 60)
        print("GENERATING INTERACTIVE PLOTS")
        print("-" * 60)

        if self.iv_voltages is None:
            raise ValueError("No simulation data available for plotting!")

        # I-V Characteristics Plot
        self._create_iv_plot()

        # Quantum Efficiency Plot
        if self.qe_values is not None:
            self._create_qe_plot()

    def _create_iv_plot(self) -> None:
        """Create interactive I-V characteristics plot."""
        fig = go.Figure()

        if self.dark_currents is not None:
            fig.add_trace(go.Scatter(
                x=self.iv_voltages,
                y=np.abs(self.dark_currents),
                mode='lines+markers',
                name='Dark Current',
                marker_color='red',
                line=dict(width=2)
            ))

        if self.light_currents is not None:
            fig.add_trace(go.Scatter(
                x=self.iv_voltages,
                y=np.abs(self.light_currents),
                mode='lines+markers',
                name='Photocurrent',
                marker_color='blue',
                line=dict(width=2)
            ))

        fig.update_layout(
            title=dict(
                text="Photodiode I-V Characteristics",
                x=0.5,
                font=dict(size=16)
            ),
            xaxis_title="Anode Voltage (V)",
            yaxis_title="Current Magnitude (A/cm)",
            yaxis_type="log",
            template="plotly_white",
            hovermode="x unified"
        )

        fig.show()

    def _create_qe_plot(self) -> None:
        """Create interactive quantum efficiency plot."""
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=self.iv_voltages,
            y=self.qe_values,
            mode='lines+markers',
            name='External QE',
            marker_color='green',
            line=dict(width=2)
        ))

        fig.update_layout(
            title=dict(
                text=f"External Quantum Efficiency @ {self.wavelength_nm} nm",
                x=0.5,
                font=dict(size=16)
            ),
            xaxis_title="Anode Voltage (V)",
            yaxis_title="External Quantum Efficiency (%)",
            yaxis_range=[0, 105],
            template="plotly_white",
            hovermode="x unified"
        )

        fig.show()

    def run_complete_simulation(self) -> None:
        """Run the complete photodiode characterization sequence."""
        print("STARTING COMPLETE PHOTODIODE SIMULATION")
        print("=" * 80)

        try:
            # Setup phase
            self.initialize_device()
            self.setup_materials()
            self.setup_doping_and_mobility()
            self.setup_physics_model()

            # Simulation phase
            self.run_dark_current_simulation()
            self.run_photocurrent_simulation()
            self.calculate_quantum_efficiency()

            # Analysis phase
            self.print_simulation_summary()
            self.create_interactive_plots()

            print("\n" + "=" * 80)
            print("✅ SIMULATION COMPLETED SUCCESSFULLY!")
            print("=" * 80)

        except Exception as e:
            print(f"\n❌ SIMULATION FAILED: {str(e)}")
            raise


def main():
    """Main execution function."""
    # Create and configure simulator
    simulator = PhotodiodeSimulator(
        device_name="photodiode",
        mesh_file="output/photodiode_mesh.msh"
    )

    # Run complete simulation
    simulator.run_complete_simulation()


if __name__ == "__main__":
    main()