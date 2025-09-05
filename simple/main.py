import devsim
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import datetime
import sys

# Import functions from the pn_sim package
from pn_sim.parameters import GLOBAL_PARAMS, SILICON_PARAMS
from pn_sim.device_builder import load_mesh_and_create_device, verify_device_structure, comprehensive_mesh_debug
from pn_sim.physics_setup import setup_physics_and_materials, setup_boundary_conditions, debug_doping_profile
from pn_sim.equilibrium_solver import solve_initial_equilibrium, debug_equilibrium_state
from pn_sim.simulation_tasks import (
    run_iv_sweep, run_spectral_sweep, run_cv_sweep_ac, calculate_qe,
    force_optical_update, comprehensive_optical_debug, debug_photocurrent_mechanism
)


def plot_results(voltages, dark_currents, light_currents, cv_voltages, capacitances,
                 wavelengths_nm, spectral_qe, light_wl, qe_bias):
    """Generates and saves summary plots."""
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Photodiode Simulation Results', fontsize=16)

    # --- I-V Plot ---
    ax1 = axs[0, 0]
    ax1.semilogy(voltages, np.abs(dark_currents), 'b-o', markersize=4, label='Dark Current')
    if light_currents is not None:
        ax1.semilogy(voltages, np.abs(light_currents), 'r-^', markersize=4, label=f'Light ({light_wl} nm)')
    ax1.set_xlabel('Voltage (V)')
    ax1.set_ylabel('Current Density (A/cm)')
    ax1.set_title('I-V Characteristics')
    ax1.grid(True, which="both", ls="--")
    ax1.legend()
    ax1.set_ylim(1e-13, 1e-1)

    # --- Spectral Response Plot ---
    ax2 = axs[0, 1]
    if len(wavelengths_nm) > 0 and len(spectral_qe) > 0:
        ax2.plot(wavelengths_nm, np.array(spectral_qe) * 100, 'g-o', markersize=4, label=f'QE at {qe_bias}V')
    ax2.set_xlabel('Wavelength (nm)')
    ax2.set_ylabel('Quantum Efficiency (%)')
    ax2.set_title('Spectral Response (Quantum Efficiency)')
    ax2.grid(True)
    ax2.legend()
    ax2.set_ylim(0, 100)

    # --- C-V Plot ---
    ax3 = axs[1, 0]
    if len(cv_voltages) > 0 and len(capacitances) > 0:
        ax3.plot(cv_voltages, np.array(capacitances) * 1e9, 'm-s', markersize=4)
    ax3.set_xlabel('Voltage (V)')
    ax3.set_ylabel('Capacitance (nF/cm²)')
    ax3.set_title('C-V Characteristics (1 MHz)')
    ax3.grid(True)

    # --- Mott-Schottky Plot ---
    ax4 = axs[1, 1]
    if len(cv_voltages) > 0 and len(capacitances) > 0:
        valid_cv = ~np.isnan(capacitances)
        inv_c_sq = 1.0 / (capacitances[valid_cv] ** 2)
        ax4.plot(cv_voltages[valid_cv], inv_c_sq, 'k-d', markersize=4, label='1/C²')
    ax4.set_xlabel('Voltage (V)')
    ax4.set_ylabel('1/C² (cm⁴/F²)')
    ax4.set_title('Mott-Schottky Plot')
    ax4.grid(True)
    ax4.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plot_filename = "output/photodiode_summary_plots.png"
    plt.savefig(plot_filename)
    print(f"\n📈 Summary plots saved to '{plot_filename}'")


def main():
    """Main execution with a comprehensive step-by-step debugging workflow."""
    os.makedirs("output", exist_ok=True)
    device = GLOBAL_PARAMS["device_name"]

    # --- STEP 1: MESH & GEOMETRY ---
    print("\n" + "=" * 80 + "\nSTEP 1: MESH LOADING AND GEOMETRY VALIDATION\n" + "=" * 80)
    load_mesh_and_create_device(device, GLOBAL_PARAMS["mesh_file"])
    verify_device_structure(device)
    comprehensive_mesh_debug(device, SILICON_PARAMS)
    print("✅ STEP 1 COMPLETE")

    # --- STEP 2: PHYSICS & MATERIALS ---
    print("\n" + "=" * 80 + "\nSTEP 2: PHYSICS AND MATERIAL SETUP\n" + "=" * 80)
    setup_physics_and_materials(device, SILICON_PARAMS, GLOBAL_PARAMS)
    debug_doping_profile(device)
    print("✅ STEP 2 COMPLETE")

    # --- STEP 3: BOUNDARY CONDITIONS ---
    print("\n" + "=" * 80 + "\nSTEP 3: BOUNDARY CONDITIONS\n" + "=" * 80)
    setup_boundary_conditions(device, SILICON_PARAMS)
    print("✅ STEP 3 COMPLETE")

    # --- STEP 4: EQUILIBRIUM SOLVE ---
    print("\n" + "=" * 80 + "\nSTEP 4: INITIAL EQUILIBRIUM SOLUTION\n" + "=" * 80)
    solve_initial_equilibrium(device)
    debug_equilibrium_state(device, SILICON_PARAMS)
    print("✅ STEP 4 COMPLETE")

    # --- STEP 5: DARK I-V SWEEP ---
    print("\n" + "=" * 80 + "\nSTEP 5: DARK I-V CHARACTERIZATION\n" + "=" * 80)
    voltages = np.linspace(-5, 1, 61)
    dark_currents = run_iv_sweep(device, voltages, p_flux=0, material_params=SILICON_PARAMS)
    print("✅ STEP 5 COMPLETE")

    # --- STEP 6: ILLUMINATED I-V SWEEP ---
    print("\n" + "=" * 80 + "\nSTEP 6: ILLUMINATED I-V CHARACTERIZATION\n" + "=" * 80)
    wavelength = 650  # nm
    flux = 1e17  # photons/cm^2/s
    light_currents = run_iv_sweep(device, voltages, p_flux=flux, material_params=SILICON_PARAMS,
                                  wavelength_nm=wavelength)
    print("✅ STEP 6 COMPLETE")

    # --- STEP 7: SPECTRAL RESPONSE (QE) SWEEP ---
    print("\n" + "=" * 80 + "\nSTEP 7: SPECTRAL RESPONSE\n" + "=" * 80)
    qe_bias = -2.0
    wavelengths_nm = np.linspace(400, 1100, 36)
    spectral_qe = run_spectral_sweep(device, wavelengths_nm, qe_bias, flux, SILICON_PARAMS, voltages, dark_currents)
    print("✅ STEP 7 COMPLETE")

    # --- STEP 8: C-V SWEEP ---
    print("\n" + "=" * 80 + "\nSTEP 8: CAPACITANCE-VOLTAGE SWEEP\n" + "=" * 80)
    cv_voltages = np.linspace(1, -5, 61)
    capacitances = run_cv_sweep_ac(device, cv_voltages, freq_hz=1e6)
    print("✅ STEP 8 COMPLETE")

    # --- FINAL RESULTS & PLOTTING ---
    print("\n" + "=" * 80 + "\nFINAL RESULTS SUMMARY\n" + "=" * 80)
    plot_results(voltages, dark_currents, light_currents,
                 cv_voltages, capacitances, wavelengths_nm, spectral_qe,
                 wavelength, qe_bias)


if __name__ == "__main__":
    main()