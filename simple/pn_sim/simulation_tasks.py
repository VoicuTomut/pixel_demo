# pn_sim/simulation_tasks.py
"""
Contains functions to perform specific simulation tasks like I-V sweeps,
C-V analysis, and quantum efficiency calculations. Includes detailed
debugging functions to verify results against physical expectations.
"""
import devsim
import numpy as np
from .utils import get_alpha_for_wavelength, get_reflectivity


def run_iv_sweep(device, voltages, p_flux, material_params, wavelength_nm=None):
    """
    Performs a voltage sweep to generate an I-V curve. The core of this function
    is iterating through bias points, updating the boundary condition, and solving
    the fully coupled (Poisson + Drift-Diffusion) system.

    Args:
        device (str): The device name.
        voltages (np.array): Voltages for the sweep.
        p_flux (float): Incident photon flux (photons/cm²/s). 0 for dark sweep.
        material_params (dict): Dictionary of silicon parameters.
        wavelength_nm (float, optional): Wavelength for illuminated sweeps.

    Returns:
        np.array: Array of calculated currents in A/cm.
    """
    condition = "Illuminated" if p_flux > 0 else "Dark"
    print(f"\n--- Running I-V Sweep ({condition}) ---")

    if p_flux > 0 and wavelength_nm:
        # For illuminated sweeps, set up the optical generation model
        # G(y) = alpha * Phi_effective * exp(alpha * y)
        alpha = get_alpha_for_wavelength(wavelength_nm, material_params)
        reflectivity = get_reflectivity(wavelength_nm, material_params)
        effective_flux = p_flux * (1.0 - reflectivity)
        devsim.set_parameter(name="alpha", value=alpha)
        devsim.set_parameter(name="EffectivePhotonFlux", value=effective_flux)
        print(
            f"  Illumination: {wavelength_nm} nm, α={alpha:.1e} cm⁻¹, R={reflectivity:.2f}, Φ_eff={effective_flux:.1e} cm⁻²s⁻¹")
    else:
        # For dark sweeps, ensure optical generation is zero.
        devsim.set_parameter(name="EffectivePhotonFlux", value=0.0)

    currents = []
    for v in voltages:
        # Update the anode bias for the current sweep point
        devsim.set_parameter(name="anode_bias", value=v)
        try:
            # Solve the coupled system at this bias point
            devsim.solve(type="dc", absolute_error=1e10, relative_error=1e-10, maximum_iterations=30)
            # Total current is the sum of electron and hole currents at the contact
            current = (devsim.get_contact_current(device=device, contact="anode", equation="HoleContinuityEquation") +
                       devsim.get_contact_current(device=device, contact="anode",
                                                  equation="ElectronContinuityEquation"))
            currents.append(current)
            print(f"  V = {v:5.2f} V, I = {current:10.3e} A/cm")
        except devsim.error:
            print(f"  Convergence failed at {v:.2f}V")
            currents.append(float('nan'))

    return np.array(currents)


def run_spectral_sweep(device, wavelengths, bias, flux, material_params, v_dark, i_dark):
    """Calculates Quantum Efficiency (QE) across a range of wavelengths."""
    print(f"\n--- Running Spectral Sweep (QE vs. Wavelength) at {bias:.1f}V ---")
    qes = []
    # Interpolate to find the dark current at the specific bias voltage.
    # This is more efficient than re-solving for the dark current each time.
    dark_current_at_bias = np.interp(bias, v_dark, i_dark)

    for wl in wavelengths:
        # Run a single-point I-V simulation under illumination for the current wavelength
        light_current_point = run_iv_sweep(device, [bias], p_flux=flux, material_params=material_params,
                                           wavelength_nm=wl)
        light_current = light_current_point[0]

        if not np.isnan(light_current):
            qe = calculate_qe(dark_current_at_bias, light_current, flux, wl)
            qes.append(qe)
            print(f"  λ = {wl:.0f} nm, QE = {qe * 100:.2f}%")
        else:
            qes.append(float('nan'))
    return qes


def run_cv_sweep_ac(device, voltages, freq_hz):
    """Performs a small-signal AC simulation to get capacitance vs. voltage."""
    print(f"\n--- Running C-V Sweep at {freq_hz / 1e6:.1f} MHz ---")
    capacitances = []

    # Set up a circuit with a small-signal AC voltage source for the analysis
    devsim.circuit_element(name="V1", n1="anode_c", n2="gnd", value=0.0, acreal=1.0)
    devsim.circuit_node_alias(node="anode_c", alias="anode_bias")
    devsim.circuit_node_alias(node="gnd", alias="cathode_bias")

    for v in voltages:
        devsim.set_parameter(name="V1_real", value=v)  # Set DC bias
        devsim.solve(type="dc", absolute_error=1e10, relative_error=1e-10)
        try:
            # Perform the AC analysis at the specified frequency
            devsim.solve(type="ac", frequency=freq_hz)
            # Formula: C = Im(I_ac) / (ω * V_ac). Since V_ac=1, C = Im(I_ac) / (2πf)
            i_imag = devsim.get_circuit_node_value(solution="ac", node="V1.I_imag")
            capacitance = i_imag / (2 * np.pi * freq_hz)
            capacitances.append(capacitance)
            # Report in nF/cm^2 for readability
            print(f"  V = {v:5.2f} V, C = {capacitance * 1e9:.3f} nF/cm²")
        except devsim.error:
            print(f"  AC solve failed at {v:.2f}V")
            capacitances.append(float('nan'))

    return np.array(capacitances)


def calculate_qe(dark_current, light_current, photon_flux, wavelength_nm):
    """
    Calculates the external Quantum Efficiency (QE).
    Formula: QE = (Photocurrent_in_carriers/sec) / (Photon_flux)
               = (I_photo / q) / Φ
    """
    photocurrent = abs(light_current - dark_current)
    q = 1.602e-19  # Elementary charge
    return (photocurrent / q) / photon_flux if photon_flux > 0 else 0.0


def force_optical_update(device):
    """Forces recalculation of the optical generation model for debugging purposes."""
    print("\n--- Forcing Optical Generation Model Update ---")
    for region in ["p_region", "n_region"]:
        alpha = devsim.get_parameter(name="alpha")
        devsim.set_parameter(name="alpha", value=alpha * 1.0000001)
        devsim.set_parameter(name="alpha", value=alpha)
        gen = devsim.get_node_model_values(device=device, region=region, name="OpticalGeneration")
        print(f"  '{region}' optical generation recalculated. Peak: {np.max(gen):.2e} cm⁻³s⁻¹")


def comprehensive_optical_debug(device):
    """Provides a detailed analysis of the optical generation profile."""
    for region in ["p_region", "n_region"]:
        y = np.array(devsim.get_node_model_values(device=device, region=region, name="y"))
        gen = np.array(devsim.get_node_model_values(device=device, region=region, name="OpticalGeneration"))

        if np.max(gen) < 1e-10:
            print(f"  ⚠️ WARNING in '{region}': Optical generation is effectively zero.")
            continue

        peak_gen = np.max(gen)
        y_at_peak = y[np.argmax(gen)]
        print(f"  '{region}' peak generation of {peak_gen:.2e} occurs at y={y_at_peak:.2f} μm.")


def debug_photocurrent_mechanism(device, bias):
    """Analyzes carrier concentrations and currents if photocurrent is unexpectedly low."""
    print("\n--- Debugging Photocurrent Flow Mechanism ---")
    devsim.set_parameter(name="anode_bias", value=bias)
    devsim.solve(type="dc", absolute_error=1e10, relative_error=1e-10)

    for region in ["p_region", "n_region"]:
        electrons = devsim.get_node_model_values(device=device, region=region, name="Electrons")
        holes = devsim.get_node_model_values(device=device, region=region, name="Holes")
        print(f"  Region '{region}' Carrier Concentrations:")
        print(f"    - Electrons: [{np.min(electrons):.2e}, {np.max(electrons):.2e}] cm⁻³")
        print(f"    - Holes:     [{np.min(holes):.2e}, {np.max(holes):.2e}] cm⁻³")
