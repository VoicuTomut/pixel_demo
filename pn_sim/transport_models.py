# transport_models.py - Add this file to pn_sim directory

import devsim
import numpy as np











def force_optical_update(device):
    """Force recalculation of optical generation after parameter change."""

    for region in ["p_region", "n_region"]:
        # Delete and recreate the optical generation model
        try:
            devsim.delete_node_model(device=device, region=region, name="OpticalGeneration")
        except:
            pass

        # Recreate with current parameters
        alpha = devsim.get_parameter(name="alpha")
        flux = devsim.get_parameter(name="EffectivePhotonFlux")

        optical_eq = f"{flux} * {alpha} * exp(-{alpha} * abs(y) * 1e-4)"
        devsim.node_model(device=device, region=region, name="OpticalGeneration",
                          equation=optical_eq)
        devsim.node_model(device=device, region=region, name="OpticalGeneration:Electrons",
                          equation="0.0")
        devsim.node_model(device=device, region=region, name="OpticalGeneration:Holes",
                          equation="0.0")


def comprehensive_optical_debug(device):
    """Debug optical generation in detail."""

    print("\n  Optical Generation Analysis:")

    alpha = devsim.get_parameter(name="alpha")
    flux = devsim.get_parameter(name="EffectivePhotonFlux")

    print(f"    Global parameters:")
    print(f"      alpha = {alpha:.2e} cm^-1")
    print(f"      flux = {flux:.2e} photons/cm^2/s")

    for region in ["p_region", "n_region"]:
        try:
            gen = np.array(devsim.get_node_model_values(device=device, region=region,
                                                        name="OpticalGeneration"))
            y = np.array(devsim.get_node_model_values(device=device, region=region,
                                                      name="y"))

            # Find surface and bulk generation
            surface_idx = np.argmin(np.abs(y))
            bulk_idx = np.argmin(y)

            print(f"    {region}:")
            print(f"      Surface generation: {gen[surface_idx]:.2e} cm^-3/s")
            print(f"      Bulk generation: {gen[bulk_idx]:.2e} cm^-3/s")
            print(f"      Max generation: {np.max(gen):.2e} cm^-3/s")
            print(f"      Total generation: {np.sum(gen):.2e}")

        except Exception as e:
            print(f"      Error analyzing {region}: {e}")


# Additional helper functions needed by main.py

def calculate_qe(dark_current, light_current, photon_flux, wavelength_nm):
    """Calculate quantum efficiency."""
    q = 1.602e-19  # Elementary charge

    # Photocurrent
    photocurrent = abs(light_current - dark_current)

    # Incident photon flux to current
    incident_current = photon_flux * q

    # QE as percentage
    if incident_current > 0:
        qe = 100 * (photocurrent / incident_current)
    else:
        qe = 0.0

    return min(qe, 100.0)  # Cap at 100%


def debug_photocurrent_mechanism(device, bias_voltage):
    """Debug why photocurrent might be low."""

    print("\n  Photocurrent Debug:")

    for region in ["p_region", "n_region"]:
        try:
            # Get generation and recombination
            gen = np.array(devsim.get_node_model_values(device=device, region=region,
                                                        name="OpticalGeneration"))
            srh = np.array(devsim.get_node_model_values(device=device, region=region,
                                                        name="USRH"))
            auger = np.array(devsim.get_node_model_values(device=device, region=region,
                                                          name="UAuger"))

            net_gen = gen - srh - auger

            print(f"    {region}:")
            print(f"      Total optical generation: {np.sum(gen):.2e}")
            print(f"      Total SRH recombination: {np.sum(srh):.2e}")
            print(f"      Total Auger recombination: {np.sum(auger):.2e}")
            print(f"      Net generation: {np.sum(net_gen):.2e}")

        except Exception as e:
            print(f"      Error: {e}")


def run_iv_sweep(device, voltages, p_flux=0, material_params=None, wavelength_nm=650):
    """Run I-V sweep with optional illumination."""

    from pn_sim.physics_setup import set_optical_parameters

    if p_flux > 0 and material_params:
        alpha, refl, eff_flux = set_optical_parameters(wavelength_nm, p_flux, material_params)
        force_optical_update(device)

    currents = []

    for v in voltages:
        devsim.set_parameter(device=device, name="anode_bias", value=v)

        try:
            devsim.solve(type="dc", absolute_error=1e2, relative_error=1e-5,
                         maximum_iterations=50)

            # Get total current
            e_curr = devsim.get_contact_current(device=device, contact="anode",
                                                equation="ElectronContinuityEquation")
            h_curr = devsim.get_contact_current(device=device, contact="anode",
                                                equation="HoleContinuityEquation")
            currents.append(e_curr + h_curr)

        except:
            currents.append(float('nan'))

    return np.array(currents)


def run_cv_sweep_ac(device, voltages, frequency_hz):
    """Simple C-V sweep - returns capacitance values."""

    capacitances = []

    for v in voltages:
        devsim.set_parameter(device=device, name="anode_bias", value=v)

        try:
            # DC solve first
            devsim.solve(type="dc", absolute_error=1e2, relative_error=1e-5,
                         maximum_iterations=30)

            # Extract depletion capacitance from charge
            # Simplified approach - just track charge change
            q_anode = devsim.get_contact_charge(device=device, contact="anode",
                                                equation="PotentialEquation")

            # Estimate capacitance from charge
            # This is simplified - proper AC analysis would be better
            cap = abs(q_anode) / max(abs(v), 0.1)  # Avoid division by zero
            capacitances.append(cap)

        except:
            capacitances.append(float('nan'))

    return np.array(capacitances)


def run_spectral_sweep(device, wavelengths, bias, flux, dark_current_interp,
                       dark_v, dark_i, material_params):
    """Run spectral response sweep."""

    from pn_sim.physics_setup import set_optical_parameters

    qe_values = []

    # Set bias
    devsim.set_parameter(device=device, name="anode_bias", value=bias)

    for wl in wavelengths:
        try:
            # Set optical parameters for this wavelength
            alpha, refl, eff_flux = set_optical_parameters(wl, flux, material_params)
            force_optical_update(device)

            # Solve
            devsim.solve(type="dc", absolute_error=1e2, relative_error=1e-5,
                         maximum_iterations=30)

            # Get currents
            e_curr = devsim.get_contact_current(device=device, contact="anode",
                                                equation="ElectronContinuityEquation")
            h_curr = devsim.get_contact_current(device=device, contact="anode",
                                                equation="HoleContinuityEquation")
            light_current = e_curr + h_curr

            # Get dark current at this bias
            dark_current = np.interp(bias, dark_v, dark_i)

            # Calculate QE
            qe = calculate_qe(dark_current, light_current, flux, wl)
            qe_values.append(qe)

        except:
            qe_values.append(float('nan'))

    return qe_values


def plot_results(iv_voltages, dark_currents, light_currents, qe_spectral,
                 cv_voltages, capacitances, wavelengths, qe_values,
                 test_wavelength, qe_bias):
    """Generate visualization plots (placeholder for actual implementation)."""

    import matplotlib.pyplot as plt

    try:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # I-V curves
        ax = axes[0, 0]
        ax.semilogy(iv_voltages, np.abs(dark_currents), 'b-', label='Dark')
        ax.semilogy(iv_voltages, np.abs(light_currents), 'r-', label='Light')
        ax.set_xlabel('Voltage (V)')
        ax.set_ylabel('Current (A/cm²)')
        ax.set_title('I-V Characteristics')
        ax.legend()
        ax.grid(True)

        # Spectral response
        ax = axes[0, 1]
        if len(wavelengths) > 0 and len(qe_values) > 0:
            ax.plot(wavelengths, qe_values, 'g-o')
        ax.set_xlabel('Wavelength (nm)')
        ax.set_ylabel('Quantum Efficiency (%)')
        ax.set_title('Spectral Response')
        ax.grid(True)

        # C-V curve
        ax = axes[1, 0]
        if len(cv_voltages) > 0 and len(capacitances) > 0:
            ax.plot(cv_voltages, capacitances, 'k-')
        ax.set_xlabel('Voltage (V)')
        ax.set_ylabel('Capacitance (F/cm²)')
        ax.set_title('C-V Characteristics')
        ax.grid(True)

        # Photocurrent vs voltage
        ax = axes[1, 1]
        photocurrent = np.abs(light_currents - dark_currents)
        ax.plot(iv_voltages, photocurrent, 'm-')
        ax.set_xlabel('Voltage (V)')
        ax.set_ylabel('Photocurrent (A/cm²)')
        ax.set_title(f'Photocurrent @ {test_wavelength}nm')
        ax.grid(True)

        plt.tight_layout()
        plt.savefig('output/photodiode_results.png', dpi=150)
        plt.show()

    except Exception as e:
        print(f"    Plot generation error: {e}")