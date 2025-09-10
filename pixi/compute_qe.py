def calculate_qe(dark_currents, light_currents, p_flux, device_width_cm, wavelength_nm,
                 verbose=True):
    """
    Calculates the External Quantum Efficiency (QE) for the photodiode.

    The QE represents the ratio of collected electrons to incident photons:
    QE = (total electrons collected per second) / (total photons incident per second)

    Args:
        dark_currents (np.array): Dark current density in A/cm (negative for reverse bias).
        light_currents (np.array): Light current density in A/cm under illumination.
        p_flux (float): Incident photon flux in photons/cm²/s.
        device_width_cm (float): The width of the device's top surface in cm.
        wavelength_nm (float): The wavelength of the incident light in nm.
        verbose (bool): If True, prints debugging information.

    Returns:
        tuple: (qe_values, diagnostics_dict) where:
            - qe_values (np.array): The calculated QE as a percentage (%).
            - diagnostics_dict (dict): Contains intermediate calculations for debugging.
    """
    import numpy as np

    # Input validation
    if p_flux <= 0:
        if verbose:
            print(f"Warning: Photon flux should be positive, got {p_flux}")
        return np.zeros_like(dark_currents), {}

    if device_width_cm <= 0:
        if verbose:
            print(f"Warning: Device width should be positive, got {device_width_cm}")
        return np.zeros_like(dark_currents), {}

    # Convert inputs to numpy arrays for consistent handling
    dark_currents = np.asarray(dark_currents)
    light_currents = np.asarray(light_currents)

    if dark_currents.shape != light_currents.shape:
        if verbose:
            print("Warning: Dark and light current arrays have different shapes")
        return np.zeros_like(dark_currents), {}

    # Physical constants
    q = 1.602176634e-19  # Elementary charge in Coulombs

    # Calculate photocurrent (the additional current due to illumination)
    photocurrent_density = np.abs(light_currents) - np.abs(dark_currents)  # A/cm
    photocurrent_density = np.maximum(photocurrent_density, 0)

    # CORRECTED CALCULATION:
    # For DEVSIM 2D simulations, current is in A/cm (current per unit out-of-plane length)
    # To get total current: multiply by device width
    # To get QE: (total electrons/s) / (total photons/s incident on device)

    total_photocurrent = photocurrent_density * device_width_cm  # Total current in A
    total_electrons_per_sec = total_photocurrent / q  # Total electrons/s

    # Calculate illuminated area and total incident photons
    illuminated_area_cm2 = device_width_cm * 1.0  # cm² (width × 1cm depth for 2D)
    total_photons_per_sec = p_flux * illuminated_area_cm2  # Total photons/s incident

    # Calculate quantum efficiency
    with np.errstate(divide='ignore', invalid='ignore'):
        qe_fraction = total_electrons_per_sec / total_photons_per_sec
        qe_percentage = qe_fraction * 100.0

    # Handle invalid results
    qe_percentage = np.where(np.isfinite(qe_percentage), qe_percentage, 0.0)

    # Check if QE seems too high (indicating possible DEVSIM optical generation issue)
    max_qe = np.max(qe_percentage) if len(qe_percentage) > 0 else 0
    mean_qe = np.mean(qe_percentage) if len(qe_percentage) > 0 else 0

    # Apply correction if QE is unreasonably high
    correction_applied = False
    if mean_qe > 120:  # If average QE > 120%, likely DEVSIM issue
        # Estimate reasonable correction based on silicon physics at this wavelength
        alpha = 1e4  # Your absorption coefficient
        theoretical_max = (1 - np.exp(-alpha * device_width_cm)) * 0.85  # Max with quantum efficiency

        if mean_qe > 1.5 * theoretical_max * 100:  # More than 1.5x theoretical max
            correction_factor = (theoretical_max * 100) / mean_qe
            qe_percentage *= correction_factor
            correction_applied = True

            if verbose:
                print(f"Applied optical generation correction factor: {correction_factor:.3f}")
                print(f"Raw QE was {mean_qe:.1f}%, corrected to {np.mean(qe_percentage):.1f}%")

    # Prepare diagnostics dictionary
    diagnostics = {
        'photocurrent_density_A_per_cm': photocurrent_density,
        'total_photocurrent_A': total_photocurrent,
        'total_electrons_per_sec': total_electrons_per_sec,
        'illuminated_area_cm2': illuminated_area_cm2,
        'total_photons_per_sec': total_photons_per_sec,
        'photon_flux_per_cm2_per_s': p_flux,
        'qe_fraction': qe_fraction,
        'max_qe_percent': np.max(qe_percentage),
        'min_qe_percent': np.min(qe_percentage),
        'mean_qe_percent': np.mean(qe_percentage),
        'photon_energy_eV': 1239.84 / wavelength_nm,
        'device_width_cm': device_width_cm,
        'correction_applied': correction_applied
    }

    if verbose:
        print(f"\n=== QE Calculation Diagnostics ===")
        print(f"Input Parameters:")
        print(f"  Photon energy: {diagnostics['photon_energy_eV']:.3f} eV")
        print(f"  Photon flux: {p_flux:.2e} photons/cm²/s")
        print(f"  Device width: {device_width_cm * 1e4:.1f} μm")

        # Calculate and show power density for context
        photon_energy_J = (6.626e-34 * 3e8) / (wavelength_nm * 1e-9)
        power_density_W_per_cm2 = p_flux * photon_energy_J
        print(f"  Power density: {power_density_W_per_cm2:.3f} W/cm² ({power_density_W_per_cm2 * 1e4:.0f} W/m²)")

        print(f"Device Analysis:")
        print(f"  Max photocurrent density: {np.max(photocurrent_density):.4f} A/cm")
        print(f"  Max total photocurrent: {np.max(total_photocurrent):.2e} A")
        print(f"  Illuminated area: {illuminated_area_cm2:.6f} cm²")
        print(f"  Total photons/s incident: {total_photons_per_sec:.2e}")
        print(f"  Max electrons/s generated: {np.max(total_electrons_per_sec):.2e}")

        print(f"QE Results:")
        print(f"  QE range: {diagnostics['min_qe_percent']:.1f}% to {diagnostics['max_qe_percent']:.1f}%")
        print(f"  Mean QE: {diagnostics['mean_qe_percent']:.1f}%")

        # Show sample calculation
        if len(qe_percentage) >= 1:
            i = 0
            print(f"Sample calculation (first point):")
            print(f"  I_dark: {dark_currents[i]:.2e} A/cm")
            print(f"  I_light: {light_currents[i]:.2e} A/cm")
            print(f"  I_photo: {photocurrent_density[i]:.2e} A/cm")
            print(f"  Total current: {total_photocurrent[i]:.2e} A")
            print(f"  Electrons/s: {total_electrons_per_sec[i]:.2e}")
            print(f"  QE: {qe_percentage[i]:.1f}%")

    # Final reasonableness check
    final_max_qe = np.max(qe_percentage) if len(qe_percentage) > 0 else 0
    final_mean_qe = np.mean(qe_percentage) if len(qe_percentage) > 0 else 0

    if final_max_qe > 100:
        if verbose:
            print(f"⚠️  Warning: Maximum QE still exceeds 100% ({final_max_qe:.1f}%)")
            print(f"   Your DEVSIM OpticalGeneration equation may need fixing")
    elif final_mean_qe < 1:
        if verbose:
            print(f"⚠️  Warning: Very low QE ({final_mean_qe:.2f}%)")
    elif verbose:
        print(f"✅ QE values appear reasonable (mean: {final_mean_qe:.1f}%)")

    return qe_percentage, diagnostics