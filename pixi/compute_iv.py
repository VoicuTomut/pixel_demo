# ============================================================================
# FILE: compute_iv.py
# PURPOSE: Perform I-V sweep measurements on the photodiode
# ============================================================================

import devsim
import numpy as np


def run_iv_sweep(device, voltages, p_flux):
    """
    Performs current-voltage (I-V) sweep on the photodiode device.

    This function:
    1. Sets the photon flux (for dark or illuminated conditions)
    2. Steps through each voltage point
    3. Solves the device equations at each bias
    4. Collects the total current (electron + hole contributions)

    Args:
        device (str): Name of the device
        voltages (np.array): Array of voltage points to sweep
        p_flux (float): Photon flux in photons/cm²/s (0 for dark)

    Returns:
        np.array: Array of currents at each voltage point (A/cm)
    """
    currents = []  # Storage for current values

    # Set the global photon flux parameter for optical generation calculations
    devsim.set_parameter(name="PhotonFlux", value=p_flux)

    # Debug output for illuminated simulations
    if p_flux > 0:
        print(f"DEBUG: PhotonFlux set to {devsim.get_parameter(name='PhotonFlux')}")

    # ===== MAIN I-V SWEEP LOOP =====
    for v in voltages:
        print(f"\nSetting Anode Bias: {v:.3f} V")

        # Apply the bias voltage to the anode contact
        devsim.set_parameter(device=device, name="anode_bias", value=v)

        try:
            # Solve the coupled semiconductor equations at this bias point
            # Using relaxed tolerances and increased iterations for better convergence
            devsim.solve(
                type="dc",  # DC steady-state solution
                absolute_error=1e10,  # Absolute error tolerance (relaxed)
                relative_error=10,  # Relative error tolerance (relaxed)
                maximum_iterations=400,  # Max Newton iterations
                maximum_divergence=10  # Allow some divergence before failing
            )

            # Extract electron current contribution at anode
            e_current = devsim.get_contact_current(
                device=device,
                contact="anode",
                equation="ElectronContinuityEquation"
            )

            # Extract hole current contribution at anode
            h_current = devsim.get_contact_current(
                device=device,
                contact="anode",
                equation="HoleContinuityEquation"
            )

            # Total current is sum of electron and hole currents
            total_current = e_current + h_current
            currents.append(total_current)

            print(f"✅ V = {v:.3f} V, Current = {total_current:.4e} A/cm")

        except devsim.error as msg:
            # Handle convergence failures gracefully
            print(f"❌ CONVERGENCE FAILED at V = {v:.3f} V. Error: {msg}")
            currents.append(float('nan'))  # Mark failed point as NaN
            break  # Stop sweep if convergence fails

    # Reset bias to 0V after sweep completes
    devsim.set_parameter(device=device, name="anode_bias", value=0.0)

    return np.array(currents)
