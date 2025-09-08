
def calculate_qe(dark_currents, light_currents, p_flux, device_width_cm, wavelength_nm,
                 verbose=False):
    """
    Calculates the External Quantum Efficiency (QE) for the photodiode.

    The QE represents the ratio of collected electrons to incident photons:
    QE = (number of electrons collected per second) / (number of photons incident per second)

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

    Raises:
        ValueError: If inputs are invalid or calculation fails.
    """
    import numpy as np

    # Input validation
    if p_flux <= 0:
        raise ValueError(f"Photon flux must be positive, got {p_flux}")

    if device_width_cm <= 0:
        raise ValueError(f"Device width must be positive, got {device_width_cm}")

    if wavelength_nm <= 0:
        raise ValueError(f"Wavelength must be positive, got {wavelength_nm}")

    # Convert inputs to numpy arrays for consistent handling
    dark_currents = np.asarray(dark_currents)
    light_currents = np.asarray(light_currents)

    if dark_currents.shape != light_currents.shape:
        raise ValueError("Dark and light current arrays must have the same shape")

    # Physical constants
    q = 1.602176634e-19  # Elementary charge in Coulombs (2019 SI definition)

    # Calculate photocurrent (the additional current due to illumination)
    # For a photodiode under reverse bias, both dark and light currents are typically negative
    # The photocurrent is the difference: |I_light| - |I_dark|
    photocurrent_density = np.abs(light_currents) - np.abs(dark_currents)  # A/cm

    # Handle cases where photocurrent might be negative (unexpected)
    if np.any(photocurrent_density < 0):
        negative_indices = photocurrent_density < 0
        if verbose:
            print(f"Warning: {np.sum(negative_indices)} points have negative photocurrent")
            print(f"Setting negative photocurrents to zero")
        photocurrent_density = np.maximum(photocurrent_density, 0)

    # Convert current density to electron generation rate
    # Current = charge × (electrons/second), so electrons/second = Current / charge
    electrons_per_sec_per_cm = photocurrent_density / q  # electrons/s/cm

    # Convert to electrons per unit area (the device has finite width)
    # Total electron rate = (electrons/s/cm) × (device_width_cm)
    # But for QE calculation, we want the rate per unit illuminated area
    electrons_per_sec_per_cm2 = electrons_per_sec_per_cm / device_width_cm  # electrons/s/cm²

    # Calculate quantum efficiency
    # QE = (electrons collected per second per cm²) / (photons incident per second per cm²)
    with np.errstate(divide='ignore', invalid='ignore'):
        qe_fraction = electrons_per_sec_per_cm2 / p_flux
        qe_percentage = qe_fraction * 100.0

    # Handle division by zero or invalid results
    qe_percentage = np.where(np.isfinite(qe_percentage), qe_percentage, 0.0)

    # Physical reasonableness check
    if np.any(qe_percentage > 100.1):  # Allow small numerical errors
        max_qe = np.max(qe_percentage)
        if verbose:
            print(f"Warning: Maximum QE ({max_qe:.2f}%) exceeds 100%. This may indicate:")
            print("  - Incorrect photon flux units")
            print("  - Device geometry issues")
            print("  - Gain mechanisms (avalanche multiplication)")

    # Prepare diagnostics dictionary
    diagnostics = {
        'photocurrent_density_A_per_cm': photocurrent_density,
        'electrons_per_sec_per_cm': electrons_per_sec_per_cm,
        'electrons_per_sec_per_cm2': electrons_per_sec_per_cm2,
        'photon_flux_per_cm2_per_s': p_flux,
        'qe_fraction': qe_fraction,
        'max_qe_percent': np.max(qe_percentage),
        'min_qe_percent': np.min(qe_percentage),
        'mean_qe_percent': np.mean(qe_percentage),
        'photon_energy_eV': 1239.84 / wavelength_nm,  # hc/λ in eV
        'device_width_cm': device_width_cm
    }

    if verbose:
        print(f"\n=== QE Calculation Diagnostics ===")
        print(f"Photon energy: {diagnostics['photon_energy_eV']:.3f} eV")
        print(f"Photon flux: {p_flux:.2e} photons/cm²/s")
        print(f"Device width: {device_width_cm * 1e4:.1f} μm")
        print(f"Photocurrent range: {np.min(photocurrent_density):.2e} to {np.max(photocurrent_density):.2e} A/cm")
        print(f"QE range: {diagnostics['min_qe_percent']:.2f}% to {diagnostics['max_qe_percent']:.2f}%")
        print(f"Mean QE: {diagnostics['mean_qe_percent']:.2f}%")

        # Show a few sample calculations
        if len(qe_percentage) >= 3:
            print(f"\nSample calculations (first 3 points):")
            for i in range(3):
                print(f"  Point {i}: I_dark={dark_currents[i]:.2e}, I_light={light_currents[i]:.2e}, "
                      f"I_photo={photocurrent_density[i]:.2e}, QE={qe_percentage[i]:.2f}%")

    return qe_percentage, diagnostics


def calculate_qe_with_diagnostics(dark_currents, light_currents, p_flux, device_width_cm, wavelength_nm):
    """
    Enhanced QE calculation with detailed diagnostics for debugging.

    This version provides the same calculation as calculate_qe() but returns additional
    diagnostic information useful for troubleshooting unexpected results.

    Returns:
        tuple: (qe_values, diagnostics_dict)
    """
    import numpy as np

    # Input validation
    if p_flux <= 0:
        raise ValueError(f"Photon flux must be positive, got {p_flux}")

    # Convert inputs to numpy arrays
    dark_currents = np.asarray(dark_currents)
    light_currents = np.asarray(light_currents)

    # Physical constants
    q = 1.602176634e-19  # Elementary charge in Coulombs

    # Calculate photocurrent
    photocurrent_density = np.abs(light_currents) - np.abs(dark_currents)
    photocurrent_density = np.maximum(photocurrent_density, 0)

    # Convert to electron rate per unit area
    electrons_per_sec_per_cm = photocurrent_density / q
    electrons_per_sec_per_cm2 = electrons_per_sec_per_cm / device_width_cm

    # Calculate QE
    with np.errstate(divide='ignore', invalid='ignore'):
        qe_fraction = electrons_per_sec_per_cm2 / p_flux
        qe_percentage = qe_fraction * 100.0

    qe_percentage = np.where(np.isfinite(qe_percentage), qe_percentage, 0.0)

    # Diagnostics
    diagnostics = {
        'photocurrent_density_A_per_cm': photocurrent_density,
        'electrons_per_sec_per_cm': electrons_per_sec_per_cm,
        'electrons_per_sec_per_cm2': electrons_per_sec_per_cm2,
        'photon_flux_per_cm2_per_s': p_flux,
        'qe_fraction': qe_fraction,
        'max_qe_percent': np.max(qe_percentage) if len(qe_percentage) > 0 else 0,
        'min_qe_percent': np.min(qe_percentage) if len(qe_percentage) > 0 else 0,
        'mean_qe_percent': np.mean(qe_percentage) if len(qe_percentage) > 0 else 0,
        'photon_energy_eV': 1239.84 / wavelength_nm,
        'device_width_cm': device_width_cm
    }

    return qe_percentage, diagnostics