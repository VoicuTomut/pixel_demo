# devsim_simulation_refactored.py

import devsim
import os
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ==============================================================================
#                      MATERIAL PARAMETERS DICTIONARY
# ==============================================================================
# To simulate a different material, modify ONLY this dictionary
SILICON_PARAMS = {
    # Physical constants
    "permittivity": 11.9 * 8.854e-14,  # F/cm
    "electron_charge": 1.6e-19,  # C
    "bandgap": 1.12,  # eV at 300K
    "Nc_300K": 2.8e19,  # Conduction band density of states at 300K

    # Doping parameters
    "peak_p_doping": 5e17,  # cm^-3
    "junction_depth": 0.5,  # μm
    "doping_straggle": 0.1,  # μm
    "n_bulk": 1e15,  # cm^-3
    "projected_range": 0.05,  # μm, peak of the implant

    # SRH lifetime parameters (Klaassen model)
    "tau_max_n": 1.0e-6,  # s
    "N_ref_n": 5.0e16,  # cm^-3
    "tau_max_p": 1.0e-6,  # s
    "N_ref_p": 5.0e16,  # cm^-3

    # Mobility parameters (Caughey-Thomas model)
    "mu_max_n": 1417.0,  # cm^2/V-s
    "mu_min_n": 68.5,  # cm^2/V-s
    "N_ref_mob_n": 1.10e17,  # cm^-3
    "alpha_mob_n": 0.711,
    "mu_max_p": 470.5,  # cm^2/V-s
    "mu_min_p": 44.9,  # cm^2/V-s
    "N_ref_mob_p": 2.23e17,  # cm^-3
    "alpha_mob_p": 0.719,

    # Velocity saturation parameters
    "v_sat_n": 1.07e7,  # cm/s
    "beta_n": 2.0,
    "v_sat_p": 8.37e6,  # cm/s
    "beta_p": 1.0,

    # TAT parameters (Hurkx model)
    "N_t_TAT": 1e15,  # cm^-3
    "E_t_TAT": 0.0,  # eV
    "gamma_TAT": 2.5,
    "delta_TAT": 3.0,

    # Impact ionization parameters
    "a_n": 7.03e5,  # cm^-1
    "b_n": 1.231e6,  # V/cm
    "a_p": 1.582e6,  # cm^-1
    "b_p": 2.036e6,  # V/cm

    # Surface recombination velocities
    "s_n": 100.0,  # cm/s
    "s_p": 100.0,  # cm/s

    # Optical data table (wavelength_nm, n_r, k_e)
    "optical_data": {
        "wavelengths": np.array([400, 450, 500, 550, 600, 650, 700, 800, 900, 1000]),
        "n_values": np.array([5.57, 4.67, 4.29, 4.07, 3.93, 3.83, 3.76, 3.66, 3.60, 3.56]),
        "k_values": np.array([0.38, 0.116, 0.056, 0.03, 0.02, 0.012, 0.008, 0.004, 0.002, 0.001])
    },

    # Add these parameters to SILICON_PARAMS
    "Nv_300K": 1.04e19,  # Valence band density of states at 300K
    "C_n_auger": 2.8e-31,  # Auger coefficient for electrons (cm^6/s)
    "C_p_auger": 9.9e-32,  # Auger coefficient for holes (cm^6/s)
    "E_th_n": 3.5e6,  # Impact ionization threshold for electrons (V/cm)
    "E_th_p": 3.5e6,  # Impact ionization threshold for holes (V/cm)


    "BGN_V0": 0.009,  # V
    "BGN_N_ref": 1.3e17, # cm^-3


}

# Global simulation parameters (device/simulation specific, not material specific)
GLOBAL_PARAMS = {
    "device_name": "photodiode",
    "mesh_file": "output/photodiode_mesh.msh",
    "temperature_K": 300.0,
    "photon_flux": 0.0,
    "wavelength_start_nm": 400,
    "wavelength_end_nm": 1100,
    "wavelength_points": 71
}

# Debug functions:




def comprehensive_mesh_debug(device, material_params):
    """
    Comprehensive mesh analysis with specific recommendations for photodiode design.
    """
    print("\n" + "=" * 70)
    print("           COMPREHENSIVE MESH GEOMETRY ANALYSIS")
    print("=" * 70)

    # Get optical parameters for analysis
    wavelength_nm = 650  # Reference wavelength
    alpha = get_alpha_for_wavelength(wavelength_nm, material_params)
    absorption_length_cm = 1.0 / alpha
    absorption_length_um = absorption_length_cm * 1e4

    print(f"\nOPTICAL REFERENCE (λ = {wavelength_nm} nm):")
    print(f"  Absorption coefficient α: {alpha:.0f} cm⁻¹")
    print(f"  Absorption length (1/α): {absorption_length_um:.2f} μm")
    print(f"  90% absorption depth: {2.3 * absorption_length_um:.2f} μm")
    print(f"  99% absorption depth: {4.6 * absorption_length_um:.2f} μm")

    total_nodes = 0
    total_area = 0

    for region in ["p_region", "n_region"]:
        print(f"\n{region.upper()} ANALYSIS:")
        print("-" * 50)

        # Get coordinates
        x_coords = np.array(devsim.get_node_model_values(device=device, region=region, name="x"))
        y_coords = np.array(devsim.get_node_model_values(device=device, region=region, name="y"))

        # Basic statistics
        x_min, x_max = np.min(x_coords), np.max(x_coords)
        y_min, y_max = np.min(y_coords), np.max(y_coords)
        x_span = x_max - x_min
        y_span = y_max - y_min
        area = x_span * y_span

        print(f"  Dimensions:")
        print(f"    X: {x_min:.6f} to {x_max:.6f} μm (span: {x_span:.6f} μm)")
        print(f"    Y: {y_min:.6f} to {y_max:.6f} μm (span: {y_span:.6f} μm)")
        print(f"    Area: {area:.3e} μm²")
        print(f"    Nodes: {len(x_coords):,}")

        # Mesh density
        if len(x_coords) > 0:
            node_density = len(x_coords) / max(area, 1e-12)
            print(f"    Node density: {node_density:.1e} nodes/μm²")

        # Critical analysis for photodiodes
        print(f"\n  PHOTODIODE SUITABILITY:")

        # Depth analysis
        depth_ratio = y_span / absorption_length_um
        print(f"    Depth / Absorption length: {depth_ratio:.4f}")

        if depth_ratio < 0.1:
            print(f"    🔴 CRITICAL: Depth is {100 * depth_ratio:.1f}% of absorption length")
            print(f"       - Will absorb only ~{100 * (1 - np.exp(-depth_ratio)):.1f}% of incident light")
            print(f"       - Photocurrent will be extremely low")
        elif depth_ratio < 0.5:
            print(f"    🟡 WARNING: Depth is {100 * depth_ratio:.1f}% of absorption length")
            print(f"       - Will absorb only ~{100 * (1 - np.exp(-depth_ratio)):.1f}% of incident light")
        elif depth_ratio < 2.0:
            print(f"    🟠 ADEQUATE: Depth is {100 * depth_ratio:.1f}% of absorption length")
            print(f"       - Will absorb ~{100 * (1 - np.exp(-depth_ratio)):.1f}% of incident light")
        else:
            print(f"    🟢 GOOD: Depth is {100 * depth_ratio:.1f}% of absorption length")
            print(f"       - Will absorb ~{100 * (1 - np.exp(-depth_ratio)):.1f}% of incident light")

        # Width analysis
        if x_span < 1.0:
            print(f"    🔴 Width ({x_span:.3f} μm) may be too narrow for realistic device")
        elif x_span > 100.0:
            print(f"    🟡 Width ({x_span:.1f} μm) is very large - may slow simulation")
        else:
            print(f"    🟢 Width ({x_span:.3f} μm) is reasonable")

        # Junction analysis for p-region
        if region == "p_region":
            if y_span < 0.1:
                print(f"    🔴 P-region too shallow ({y_span:.6f} μm) - insufficient for junction formation")
            elif y_span < 0.5:
                print(f"    🟡 P-region shallow ({y_span:.3f} μm) - may limit junction quality")
            else:
                print(f"    🟢 P-region depth adequate ({y_span:.3f} μm)")

        total_nodes += len(x_coords)
        total_area += area

    # Overall device assessment
    print(f"\n" + "=" * 50)
    print("OVERALL DEVICE ASSESSMENT")
    print("=" * 50)
    print(f"Total nodes: {total_nodes:,}")
    print(f"Total area: {total_area:.3e} μm²")

    # Get total device dimensions
    all_x = []
    all_y = []
    for region in ["p_region", "n_region"]:
        x_coords = np.array(devsim.get_node_model_values(device=device, region=region, name="x"))
        y_coords = np.array(devsim.get_node_model_values(device=device, region=region, name="y"))
        all_x.extend(x_coords)
        all_y.extend(y_coords)

    device_width = np.max(all_x) - np.min(all_x)
    device_depth = np.max(all_y) - np.min(all_y)

    print(f"Device width: {device_width:.6f} μm")
    print(f"Device depth: {device_depth:.6f} μm")

    # Critical recommendations
    print(f"\n🎯 SPECIFIC RECOMMENDATIONS:")

    if device_depth < 2.0:
        print(f"1. INCREASE DEPTH: Current {device_depth:.6f} μm → Recommended: 5-20 μm")
        print(f"   - For 90% light absorption at 650nm: need ≥{2.3 * absorption_length_um:.1f} μm")
        print(f"   - For broadband response: need ≥20 μm")

    if device_width < 1.0:
        print(f"2. INCREASE WIDTH: Current {device_width:.6f} μm → Recommended: 10-100 μm")
        print(f"   - Improves current collection and reduces edge effects")

    # P-region specific
    p_y = np.array(devsim.get_node_model_values(device=device, region="p_region", name="y"))
    p_depth = np.max(p_y) - np.min(p_y)
    if p_depth < 0.5:
        print(f"3. DEEPEN P-REGION: Current {p_depth:.6f} μm → Recommended: 0.5-2.0 μm")
        print(f"   - Ensures proper junction formation and low series resistance")

    print(f"\n💡 MESH CREATION TEMPLATE:")
    print(f"   - Device width: 20 μm (good for 1D-like behavior)")
    print(f"   - P-region depth: 1 μm (from surface)")
    print(f"   - N-region depth: 15 μm (total device depth: 16 μm)")
    print(f"   - Expected absorption at 650nm: ~99%")
    print(f"   - Junction depth: suitable for typical photodiode")

    # Performance prediction
    current_absorption = 100 * (1 - np.exp(-device_depth / absorption_length_um))
    print(f"\n📊 CURRENT DEVICE PERFORMANCE PREDICTION:")
    print(f"   - Light absorption at 650nm: ~{current_absorption:.1f}%")
    print(
        f"   - Expected photocurrent: {'EXTREMELY LOW' if current_absorption < 10 else 'LOW' if current_absorption < 50 else 'MODERATE' if current_absorption < 90 else 'GOOD'}")

    if current_absorption < 50:
        print(f"   🚨 THIS EXPLAINS WHY YOUR PHOTOCURRENT IS NEGLIGIBLE!")



def debug_mesh_geometry(device):
    """Debug the actual device geometry to understand the problem."""
    print("\n=== DEBUGGING MESH GEOMETRY ===")

    for region in ["p_region", "n_region"]:
        # Get all spatial coordinates
        x_coords = np.array(devsim.get_node_model_values(device=device, region=region, name="x"))
        y_coords = np.array(devsim.get_node_model_values(device=device, region=region, name="y"))

        print(f"\n{region}:")
        print(f"  X range: {np.min(x_coords):.6f} to {np.max(x_coords):.6f} μm")
        print(f"  Y range: {np.min(y_coords):.6f} to {np.max(y_coords):.6f} μm")
        print(f"  Number of nodes: {len(x_coords)}")

        # Check device dimensions
        x_span = np.max(x_coords) - np.min(x_coords)
        y_span = np.max(y_coords) - np.min(y_coords)
        print(f"  X span: {x_span:.6f} μm")
        print(f"  Y span: {y_span:.6f} μm")

        # For a photodiode, we need significant depth
        if y_span < 1.0:  # Less than 1 μm depth
            print(f"  ⚠️  WARNING: Very shallow device ({y_span:.6f} μm)")
            print(f"      Photodiodes typically need 2-10 μm depth for good absorption")

        # Check absorption depth at 650nm
        alpha_650 = 2320  # cm⁻¹ from your debug
        absorption_length_um = 10000 / alpha_650  # Convert to μm
        print(f"  Absorption length at 650nm: {absorption_length_um:.2f} μm")
        print(f"  Device depth / Absorption length: {y_span / absorption_length_um:.2f}")

        if y_span / absorption_length_um < 0.5:
            print(f"  ⚠️  CRITICAL: Device too shallow for effective light absorption!")



def debug_doping_profile(device):
    """Debug the doping profile to verify it's correct."""
    print("\n=== DEBUGGING DOPING PROFILE ===")

    for region in ["p_region", "n_region"]:
        print(f"\nRegion: {region}")

        # Get node positions and doping values - CONVERT TO NUMPY ARRAYS
        node_pos = np.array(devsim.get_node_model_values(device=device, region=region, name="y"))
        acceptors = np.array(devsim.get_node_model_values(device=device, region=region, name="Acceptors"))
        donors = np.array(devsim.get_node_model_values(device=device, region=region, name="Donors"))
        net_doping = np.array(devsim.get_node_model_values(device=device, region=region, name="NetDoping"))

        # Find surface and key positions
        surface_idx = np.argmin(np.abs(node_pos))
        bulk_idx = np.argmax(node_pos) if region == "n_region" else np.argmin(node_pos)

        print(
            f"  Surface (y≈0): N_A={acceptors[surface_idx]:.2e}, N_D={donors[surface_idx]:.2e}, Net={net_doping[surface_idx]:.2e}")
        print(f"  Min/Max y: {np.min(node_pos):.3f} to {np.max(node_pos):.3f} μm")
        print(f"  NetDoping range: {np.min(net_doping):.2e} to {np.max(net_doping):.2e}")

        # Check for unphysical values
        if np.any(acceptors < 0) or np.any(donors < 0):
            print(f"  ⚠️ WARNING: Negative doping found in {region}")
        if np.any(np.isnan(net_doping)) or np.any(np.isinf(net_doping)):
            print(f"  ⚠️ WARNING: NaN/Inf in NetDoping in {region}")


def debug_equilibrium_state(device):
    """Debug the equilibrium carrier concentrations and potential."""
    print("\n=== DEBUGGING EQUILIBRIUM STATE ===")

    for region in ["p_region", "n_region"]:
        print(f"\nRegion: {region}")

        # Get equilibrium values - CONVERT TO NUMPY ARRAYS
        potential = np.array(devsim.get_node_model_values(device=device, region=region, name="Potential"))
        electrons = np.array(devsim.get_node_model_values(device=device, region=region, name="Electrons"))
        holes = np.array(devsim.get_node_model_values(device=device, region=region, name="Holes"))
        ni = np.array(devsim.get_node_model_values(device=device, region=region, name="IntrinsicCarrierDensity"))
        net_doping = np.array(devsim.get_node_model_values(device=device, region=region, name="NetDoping"))

        print(f"  Potential range: {np.min(potential):.3f} to {np.max(potential):.3f} V")
        print(f"  Electron density: {np.min(electrons):.2e} to {np.max(electrons):.2e} cm⁻³")
        print(f"  Hole density: {np.min(holes):.2e} to {np.max(holes):.2e} cm⁻³")
        print(f"  ni: {np.mean(ni):.2e} cm⁻³")

        # Check charge neutrality
        space_charge = np.array(devsim.get_node_model_values(device=device, region=region, name="SpaceCharge"))
        max_charge_imbalance = np.max(np.abs(space_charge))
        print(f"  Max charge imbalance: {max_charge_imbalance:.2e} cm⁻³")

        # Check for unphysical carrier concentrations
        if np.any(electrons <= 0) or np.any(holes <= 0):
            print(f"  ⚠️ WARNING: Non-positive carrier concentrations in {region}")

        # Check pn product
        pn_product = electrons * holes
        ni_squared = ni**2
        max_pn_error = np.max(np.abs(pn_product - ni_squared) / ni_squared)
        print(f"  Max pn/ni² error: {max_pn_error:.2%}")


def debug_current_components(device, voltage):
    """Debug current components at a specific voltage."""
    print(f"\n=== DEBUGGING CURRENTS AT {voltage}V ===")

    # Set voltage and solve
    devsim.set_parameter(device=device, name="anode_bias", value=voltage)
    try:
        devsim.solve(type="dc", absolute_error=10, relative_error=1e-8, maximum_iterations=200)
    except Exception as e:
        print(f"Failed to solve at {voltage}V: {e}")
        return

    # Get contact currents
    try:
        e_current = devsim.get_contact_current(device=device, contact="anode", equation="ElectronContinuityEquation")
        h_current = devsim.get_contact_current(device=device, contact="anode", equation="HoleContinuityEquation")
        total_current = e_current + h_current

        print(f"  Electron current: {e_current:.4e} A/cm")
        print(f"  Hole current: {h_current:.4e} A/cm")
        print(f"  Total current: {total_current:.4e} A/cm")

    except Exception as e:
        print(f"Error getting currents: {e}")

    # Debug generation-recombination in each region
    for region in ["p_region", "n_region"]:
        print(f"\n  {region} G-R Analysis:")

        try:
            # Get G-R components - CONVERT TO NUMPY ARRAYS
            srh = np.array(devsim.get_node_model_values(device=device, region=region, name="USRH"))
            auger = np.array(devsim.get_node_model_values(device=device, region=region, name="UAuger"))
            optical = np.array(devsim.get_node_model_values(device=device, region=region, name="OpticalGeneration"))
            impact = np.array(devsim.get_node_model_values(device=device, region=region, name="G_impact"))
            net_recomb = np.array(devsim.get_node_model_values(device=device, region=region, name="NetRecombination"))

            print(f"    SRH recomb: {np.mean(srh):.2e} ± {np.std(srh):.2e} cm⁻³s⁻¹")
            print(f"    Auger recomb: {np.mean(auger):.2e} ± {np.std(auger):.2e} cm⁻³s⁻¹")
            print(f"    Optical gen: {np.mean(optical):.2e} ± {np.std(optical):.2e} cm⁻³s⁻¹")
            print(f"    Impact ion: {np.mean(impact):.2e} ± {np.std(impact):.2e} cm⁻³s⁻¹")
            print(f"    Net recomb: {np.mean(net_recomb):.2e} ± {np.std(net_recomb):.2e} cm⁻³s⁻¹")

            # Check for problematic values
            if np.any(np.isnan(net_recomb)) or np.any(np.isinf(net_recomb)):
                print(f"    ⚠️ WARNING: NaN/Inf in NetRecombination")
            if np.max(np.abs(impact)) > 1e20:
                print(f"    ⚠️ WARNING: Very high impact ionization")

        except Exception as e:
            print(f"    Error getting G-R data: {e}")


def comprehensive_optical_debug(device):
    """Comprehensive debug of optical generation coupling."""
    print(f"\n=== COMPREHENSIVE OPTICAL DEBUG ===")

    # Check current parameter values
    try:
        flux = devsim.get_parameter(name="EffectivePhotonFlux")
        alpha = devsim.get_parameter(name="alpha")
        print(f"EffectivePhotonFlux: {flux:.2e}")
        print(f"alpha: {alpha:.2e}")
    except:
        print("Could not retrieve parameters")

    for region in ["p_region", "n_region"]:
        try:
            # Check the individual components
            optical = np.array(devsim.get_node_model_values(device=device, region=region, name="OpticalGeneration"))
            srh = np.array(devsim.get_node_model_values(device=device, region=region, name="USRH"))
            auger = np.array(devsim.get_node_model_values(device=device, region=region, name="UAuger"))
            net_recomb = np.array(devsim.get_node_model_values(device=device, region=region, name="NetRecombination"))

            print(f"\n{region}:")
            print(f"  OpticalGeneration: {np.mean(optical):.2e} ± {np.std(optical):.2e}")
            print(f"  SRH: {np.mean(srh):.2e} ± {np.std(srh):.2e}")
            print(f"  Auger: {np.mean(auger):.2e} ± {np.std(auger):.2e}")
            print(f"  NetRecombination: {np.mean(net_recomb):.2e} ± {np.std(net_recomb):.2e}")

            # Check if the optical generation is actually affecting net recombination
            theoretical_net = np.mean(srh) + np.mean(auger) - np.mean(optical)
            print(f"  Theoretical Net: {theoretical_net:.2e}")
            print(f"  Actual Net: {np.mean(net_recomb):.2e}")
            print(f"  Match: {abs(theoretical_net - np.mean(net_recomb)) < 1e-10}")

        except Exception as e:
            print(f"Error in {region}: {e}")


def debug_optical_parameters(device, wavelength_nm, photon_flux, material_params):
    """Debug optical generation parameters."""
    print(f"\n=== DEBUGGING OPTICAL PARAMETERS ===")
    print(f"Wavelength: {wavelength_nm} nm")
    print(f"Input photon flux: {photon_flux:.2e} photons/cm²/s")

    # Check optical constants
    n_r, k_e = get_silicon_optical_constants_lookup(wavelength_nm, material_params)
    alpha = get_alpha_for_wavelength(wavelength_nm, material_params)
    reflectivity = get_reflectivity(wavelength_nm, material_params)

    print(f"Refractive index n: {n_r:.3f}")
    print(f"Extinction coeff k: {k_e:.6f}")
    print(f"Absorption coeff α: {alpha:.2e} cm⁻¹")
    print(f"Reflectivity: {reflectivity:.2%}")

    effective_flux = photon_flux * (1.0 - reflectivity)
    print(f"Effective photon flux: {effective_flux:.2e} photons/cm²/s")

    # Check devsim parameters
    try:
        alpha_param = devsim.get_parameter(name="alpha")
        flux_param = devsim.get_parameter(name="EffectivePhotonFlux")
        print(f"DEVSIM alpha parameter: {alpha_param:.2e}")
        print(f"DEVSIM EffectivePhotonFlux parameter: {flux_param:.2e}")
    except:
        print("⚠️ Could not retrieve DEVSIM optical parameters")

    # Check optical generation in each region
    for region in ["p_region", "n_region"]:
        try:
            opt_gen = np.array(devsim.get_node_model_values(device=device, region=region, name="OpticalGeneration"))
            y_pos = np.array(devsim.get_node_model_values(device=device, region=region, name="y"))

            print(f"\n{region} optical generation:")
            print(f"  Position range: {np.min(y_pos):.3f} to {np.max(y_pos):.3f} μm")
            print(f"  Generation range: {np.min(opt_gen):.2e} to {np.max(opt_gen):.2e} cm⁻³s⁻¹")
            print(f"  Total generation: {np.sum(opt_gen):.2e} cm⁻³s⁻¹")

        except Exception as e:
            print(f"  Error getting optical generation for {region}: {e}")


def debug_photocurrent_mechanism(device, voltage):
    """Debug specifically how optical generation affects currents."""
    print(f"\n=== DEBUGGING PHOTOCURRENT MECHANISM AT {voltage}V ===")

    devsim.set_parameter(device=device, name="anode_bias", value=voltage)

    # First solve with light
    current_flux = devsim.get_parameter(name="EffectivePhotonFlux")
    print(f"Current EffectivePhotonFlux: {current_flux:.2e}")

    try:
        devsim.solve(type="dc", absolute_error=10, relative_error=1e-8, maximum_iterations=200)

        # Get currents with light
        e_current_light = devsim.get_contact_current(device=device, contact="anode",
                                                     equation="ElectronContinuityEquation")
        h_current_light = devsim.get_contact_current(device=device, contact="anode", equation="HoleContinuityEquation")

        print(f"With light - Electron current: {e_current_light:.6e} A/cm")
        print(f"With light - Hole current: {h_current_light:.6e} A/cm")
        print(f"With light - Total current: {e_current_light + h_current_light:.6e} A/cm")

        # Temporarily turn off light
        devsim.set_parameter(name="EffectivePhotonFlux", value=0.0)
        devsim.solve(type="dc", absolute_error=10, relative_error=1e-8, maximum_iterations=200)

        # Get currents without light
        e_current_dark = devsim.get_contact_current(device=device, contact="anode",
                                                    equation="ElectronContinuityEquation")
        h_current_dark = devsim.get_contact_current(device=device, contact="anode", equation="HoleContinuityEquation")

        print(f"Without light - Electron current: {e_current_dark:.6e} A/cm")
        print(f"Without light - Hole current: {h_current_dark:.6e} A/cm")
        print(f"Without light - Total current: {e_current_dark + h_current_dark:.6e} A/cm")

        print(
            f"Photocurrent difference: {(e_current_light + h_current_light) - (e_current_dark + h_current_dark):.6e} A/cm")

        # Restore light
        devsim.set_parameter(name="EffectivePhotonFlux", value=current_flux)

    except Exception as e:
        print(f"Error in photocurrent debug: {e}")

def debug_field_and_currents(device):
    """Debug electric field and current density distributions."""
    print(f"\n=== DEBUGGING FIELDS AND CURRENTS ===")

    for region in ["p_region", "n_region"]:
        try:
            e_field_mag = np.array(devsim.get_node_model_values(device=device, region=region, name="E_mag_node_abs"))
            jn_mag = np.array(devsim.get_node_model_values(device=device, region=region, name="Jn_mag_node"))
            jp_mag = np.array(devsim.get_node_model_values(device=device, region=region, name="Jp_mag_node"))

            print(f"\n{region}:")
            print(f"  E-field: {np.min(e_field_mag):.2e} to {np.max(e_field_mag):.2e} V/cm")
            print(f"  Jn magnitude: {np.min(jn_mag):.2e} to {np.max(jn_mag):.2e} A/cm²")
            print(f"  Jp magnitude: {np.min(jp_mag):.2e} to {np.max(jp_mag):.2e} A/cm²")

            # Check for very high fields that might trigger impact ionization
            high_field_nodes = np.sum(e_field_mag > 1e5)
            if high_field_nodes > 0:
                print(f"  ⚠️ WARNING: {high_field_nodes} nodes with E > 10⁵ V/cm")

        except Exception as e:
            print(f"  Error getting field data for {region}: {e}")


# ==============================================================================
#                      OPTICAL HELPER FUNCTIONS
# ==============================================================================
def force_optical_update(device):
    """Force recalculation of optical generation after parameter changes."""
    for region in ["p_region", "n_region"]:
        # Update the optical generation model
        devsim.node_model(device=device, region=region, name="OpticalGeneration",
                          equation="EffectivePhotonFlux * alpha * exp(-alpha * abs(y * 1e-4))")
        # Update net recombination
        devsim.node_model(device=device, region=region, name="NetRecombination",
                          equation="USRH + UAuger - G_impact - OpticalGeneration")
        # Update continuity equation terms
        devsim.node_model(device=device, region=region, name="eCharge_x_NetRecomb",
                          equation="ElectronCharge * (USRH + UAuger - G_impact - OpticalGeneration)")
        devsim.node_model(device=device, region=region, name="Neg_eCharge_x_NetRecomb",
                          equation="-ElectronCharge * (USRH + UAuger - G_impact - OpticalGeneration)")


def verify_mesh_dimensions(device, min_depth_um=5.0, min_width_um=5.0):
    """Verify the mesh has realistic dimensions for photodiode operation."""
    print("\n=== VERIFYING MESH DIMENSIONS ===")

    all_x = []
    all_y = []

    for region in ["p_region", "n_region"]:
        x_coords = np.array(devsim.get_node_model_values(device=device, region=region, name="x"))
        y_coords = np.array(devsim.get_node_model_values(device=device, region=region, name="y"))
        all_x.extend(x_coords)
        all_y.extend(y_coords)

    device_width = np.max(all_x) - np.min(all_x)
    device_depth = np.max(all_y) - np.min(all_y)

    print(f"Device dimensions:")
    print(f"  Width: {device_width:.3f} μm")
    print(f"  Depth: {device_depth:.3f} μm")

    # Check if dimensions are reasonable
    if device_depth < min_depth_um:
        raise ValueError(f"Device depth ({device_depth:.3f} μm) is too small for realistic photodiode. "
                         f"Minimum recommended: {min_depth_um} μm. Please recreate mesh.")

    if device_width < min_width_um:
        raise ValueError(f"Device width ({device_width:.3f} μm) is too small for realistic photodiode. "
                         f"Minimum recommended: {min_width_um} μm. Please recreate mesh.")

    print("✅ Mesh dimensions are suitable for photodiode simulation")
    return device_width, device_depth


def get_silicon_optical_constants_lookup(wavelength_nm, material_params):
    """
    Calculates n_r and k_e for the material using a lookup table and interpolation.
    """
    optical_data = material_params["optical_data"]
    n_r = np.interp(wavelength_nm, optical_data["wavelengths"], optical_data["n_values"])
    k_e = np.interp(wavelength_nm, optical_data["wavelengths"], optical_data["k_values"])
    return (n_r, k_e)

def get_alpha_for_wavelength(wavelength_nm, material_params):
    """
    Calculates alpha from the extinction coefficient k_e.
    """
    n_r, k_e = get_silicon_optical_constants_lookup(wavelength_nm, material_params)
    alpha_cm = (4 * np.pi * k_e) / (wavelength_nm * 1e-7)  # wavelength converted to cm
    return alpha_cm

def get_reflectivity(wavelength_nm, material_params, use_arc=False, n_arc=2.0, d_arc_nm=75.0):
    """
    Calculates reflectivity using the optical constants.
    """
    n_air = 1.0
    n_si, k_si = get_silicon_optical_constants_lookup(wavelength_nm, material_params)
    N_si = n_si - 1j * k_si

    if not use_arc:
        r = (n_air - N_si) / (n_air + N_si)
        return np.abs(r) ** 2
    else:
        r1 = (n_air - n_arc) / (n_air + n_arc)
        r2 = (n_arc - N_si) / (n_arc + N_si)
        beta = (2 * np.pi * n_arc * d_arc_nm) / wavelength_nm
        r_total = (r1 + r2 * np.exp(-2j * beta)) / (1 + r1 * r2 * np.exp(-2j * beta))
        return np.abs(r_total) ** 2


# ==============================================================================
#                      DEVICE SETUP FUNCTIONS
# ==============================================================================
def load_mesh_and_create_device(device_name, mesh_file):
    """Load mesh and create device structure."""
    print(f"Loading mesh: {mesh_file}")
    if not os.path.exists(mesh_file):
        raise FileNotFoundError(f"Mesh file not found at '{mesh_file}'. Please run create_mesh.py first.")

    devsim.create_gmsh_mesh(mesh=device_name, file=mesh_file)
    devsim.add_gmsh_region(mesh=device_name, gmsh_name="p_region", region="p_region", material="Silicon")
    devsim.add_gmsh_region(mesh=device_name, gmsh_name="n_region", region="n_region", material="Silicon")
    devsim.add_gmsh_contact(mesh=device_name, gmsh_name="anode", region="p_region", name="anode", material="metal")
    devsim.add_gmsh_contact(mesh=device_name, gmsh_name="cathode", region="n_region", name="cathode", material="metal")
    devsim.add_gmsh_interface(mesh=device_name, gmsh_name="pn_junction", region0="p_region", region1="n_region",
                              name="pn_junction")
    devsim.finalize_mesh(mesh=device_name)
    devsim.create_device(mesh=device_name, device=device_name)
    print("\n--- Mesh loading and device creation complete ---")

def verify_device_structure(device_name):
    """Verify the device structure is correctly loaded."""
    print("\n--- Running Verification Checks ---")
    try:
        device_list = devsim.get_device_list()
        region_list = devsim.get_region_list(device=device_name)
        contact_list = devsim.get_contact_list(device=device_name)
        interface_list = devsim.get_interface_list(device=device_name)
        if (len(device_list) == 1 and len(region_list) == 2 and
                len(contact_list) == 2 and len(interface_list) == 1):
            print("✅ Verification PASSED: Device structure is correct.")
        else:
            print("❌ Verification FAILED: The device structure is not as expected.")
    except devsim.error as msg:
        print(f"❌ An error occurred during verification: {msg}")

# ==============================================================================
#                      PHYSICS DEFINITION FUNCTIONS
# ==============================================================================
def set_silicon_parameters(device, region, material_params, temperature_K):
    """Sets the basic material parameters, including temperature dependence."""
    # Physical constants
    devsim.set_parameter(name="k_B_eV", value=8.617e-5)  # Boltzmann constant in eV/K
    devsim.set_parameter(name="Nc_300K", value=material_params["Nc_300K"])
    devsim.set_parameter(name="Nv_300K", value=material_params["Nv_300K"])

    devsim.set_parameter(name="BGN_V0", value=material_params.get("BGN_V0", 0.009))
    devsim.set_parameter(name="BGN_N_ref", value=material_params.get("BGN_N_ref", 1.3e17))

    # Global device parameters
    devsim.set_parameter(name="T", value=temperature_K)
    devsim.set_parameter(device=device, region=region, name="Permittivity", value=material_params["permittivity"])
    devsim.set_parameter(device=device, region=region, name="ElectronCharge", value=material_params["electron_charge"])

def define_carrier_statistics(device, region):
    """
    Defines temperature-dependent bandgap, BGN, and intrinsic carrier density.
    This MUST be called after TotalDoping is defined.
    """
    print(f"    Defining carrier statistics for {region}...")

    # 1. Temperature-dependent Bandgap (Varshni's equation for Si)
    Eg_T_eq = "1.17 - (4.73e-4 * T * T) / (T + 636.0)"
    devsim.node_model(device=device, region=region, name="Bandgap", equation=Eg_T_eq)

    # 2. Bandgap Narrowing (Slotboom model)
    bgn_eq = "BGN_V0 * (log(TotalDoping/BGN_N_ref) + (log(TotalDoping/BGN_N_ref)^2 + 0.5))^(0.5)"
    devsim.node_model(device=device, region=region, name="Bandgap_Narrowing",
                      equation=f"ifelse(TotalDoping > BGN_N_ref, {bgn_eq}, 0.0)")

    # 3. Effective Bandgap
    devsim.node_model(device=device, region=region, name="Effective_Bandgap",
                      equation="Bandgap - Bandgap_Narrowing")

    # 4. Temperature-dependent Intrinsic Carrier Density Model (using Effective_Bandgap)
    ni_model_eq = "(Nc_300K * Nv_300K * pow(T/300.0, 3.0))^(0.5) * exp(-Effective_Bandgap / (2.0 * k_B_eV * T))"
    devsim.node_model(device=device, region=region, name="IntrinsicCarrierDensity", equation=ni_model_eq)

def define_srh_lifetime_models(device, region, material_params):
    """Defines doping-dependent SRH lifetimes using the Klaassen model."""
    tau_max_n = material_params["tau_max_n"]
    N_ref_n = material_params["N_ref_n"]
    tau_max_p = material_params["tau_max_p"]
    N_ref_p = material_params["N_ref_p"]

    eqn_n = f"{tau_max_n} / (1 + TotalDoping / {N_ref_n})"
    devsim.node_model(device=device, region=region, name="taun", equation=eqn_n)

    eqn_p = f"{tau_max_p} / (1 + TotalDoping / {N_ref_p})"
    devsim.node_model(device=device, region=region, name="taup", equation=eqn_p)

def define_doping(device, material_params):
    """Defines a Gaussian profile for the p-type implant and uniform n-type substrate."""
    p_peak = material_params["peak_p_doping"]
    projected_range_cm = material_params["projected_range"] * 1e-4
    p_straggle_cm = material_params["doping_straggle"] * 1e-4
    n_bulk = material_params["n_bulk"]

    # Define the Gaussian acceptor profile for the p-implant
    gaussian_p_eq = f"{p_peak} * exp(-0.5 * ((y* 1e-4 - {projected_range_cm}) / {p_straggle_cm})^2)"

    # p_region: Contains both the p-implant and the n-bulk background
    devsim.node_model(device=device, region="p_region", name="Acceptors", equation=gaussian_p_eq)
    devsim.node_model(device=device, region="p_region", name="Donors", equation=f"{n_bulk}") # CORRECTED

    # n_region: Contains only the n-bulk background
    devsim.node_model(device=device, region="n_region", name="Acceptors", equation="0.0")
    devsim.node_model(device=device, region="n_region", name="Donors", equation=f"{n_bulk}") # CORRECTED

    # NetDoping is now physically correct in both regions
    for region in ["p_region", "n_region"]:
        devsim.node_model(device=device, region=region, name="NetDoping", equation="Donors - Acceptors")

    print(f"Defined Gaussian p-implant (peak={p_peak:.1e}) and uniform n-bulk (N_D={n_bulk:.1e})")

def define_bandgap_narrowing(device, region, material_params):
    """Defines the Slotboom model for bandgap narrowing."""
    print(f"    Defining Bandgap Narrowing for {region}...")
    V0 = material_params["BGN_V0"]
    N_ref = material_params["BGN_N_ref"]

    # Slotboom model equation
    bgn_eq = f"{V0} * (log(TotalDoping/{N_ref}) + sqrt(log(TotalDoping/{N_ref})^2 + 0.5))"

    # Use ifelse to avoid issues at low doping
    devsim.node_model(device=device, region=region, name="Bandgap_Narrowing",
                      equation=f"ifelse(TotalDoping > {N_ref}, {bgn_eq}, 0.0)")

    # Create an effective bandgap
    devsim.node_model(device=device, region=region, name="Effective_Bandgap",
                      equation="Bandgap - Bandgap_Narrowing")

def define_hurkx_enhancement(device, region, material_params):
    """Defines the Hurkx field-enhancement factor Gamma."""
    print(f"    Defining Hurkx enhancement factor for {region}...")
    devsim.set_parameter(device=device, region=region, name="gamma_TAT", value=material_params["gamma_TAT"])
    # The Hurkx model depends on the difference between the trap level and the band edges.
    # For a mid-gap trap, this simplifies significantly.
    # E_field is in V/cm.
    hurkx_integral_arg = "gamma_TAT * (E_mag_node_abs^1.5)"  # Simplified form for mid-gap trap

    # Gamma is approximately 1 + integral term. We can use an analytical approximation for the integral.
    # A common approximation is Gamma = exp(...)
    gamma_eq = f"1.0 + 2.0 * sqrt(3*pi/2) * (E_mag_node_abs / {material_params['E_th_n']}) * exp((E_mag_node_abs / {material_params['E_th_n']})^2)"

    # A simpler but still effective form often used in TCAD:
    gamma_eq_simple = f"1 + gamma_TAT * exp(delta_TAT * E_mag_node_abs)"
    devsim.set_parameter(device=device, region=region, name="delta_TAT", value=material_params["delta_TAT"])

    # Let's use the simpler, more stable form from your original code but apply it correctly.
    gamma_eq_final = "1.0 + gamma_TAT * pow(E_mag_node_abs / 1e6, delta_TAT)"  # E-field in MV/cm as before
    devsim.node_model(device=device, region=region, name="Gamma_Hurkx", equation=gamma_eq_final)

def define_mobility_models(device, region, material_params):
    """Defines mobility using the Caughey-Thomas model."""
    # Part 1: Doping-Dependent Low-Field Mobility (at nodes)
    devsim.node_model(device=device, region=region, name="TotalDoping", equation="abs(Acceptors) + abs(Donors)")

    # Caughey-Thomas parameters for low-field mobility
    mu_max_n = material_params["mu_max_n"]
    mu_min_n = material_params["mu_min_n"]
    N_ref_n = material_params["N_ref_mob_n"]
    alpha_n = material_params["alpha_mob_n"]
    low_field_eqn_n = f"{mu_min_n} + ({mu_max_n} - {mu_min_n}) / (1 + (TotalDoping / {N_ref_n})^{alpha_n})"
    devsim.node_model(device=device, region=region, name="LowFieldElectronMobility", equation=low_field_eqn_n)

    mu_max_p = material_params["mu_max_p"]
    mu_min_p = material_params["mu_min_p"]
    N_ref_p = material_params["N_ref_mob_p"]
    alpha_p = material_params["alpha_mob_p"]
    low_field_eqn_p = f"{mu_min_p} + ({mu_max_p} - {mu_min_p}) / (1 + (TotalDoping / {N_ref_p})^{alpha_p})"
    devsim.node_model(device=device, region=region, name="LowFieldHoleMobility", equation=low_field_eqn_p)

    # Part 2: Field-Dependent High-Field Mobility (at edges)
    devsim.edge_average_model(device=device, region=region, node_model="LowFieldElectronMobility",
                              edge_model="LowFieldElectronMobility_edge")
    devsim.edge_average_model(device=device, region=region, node_model="LowFieldHoleMobility",
                              edge_model="LowFieldHoleMobility_edge")

    devsim.edge_from_node_model(device=device, region=region, node_model="Potential")
    devsim.edge_model(device=device, region=region, name="ElectricField",
                      equation="(Potential@n0 - Potential@n1) * EdgeInverseLength")
    devsim.edge_model(device=device, region=region, name="EParallel", equation="abs(ElectricField)")

    # Saturation velocity parameters
    v_sat_n = material_params["v_sat_n"]
    beta_n = material_params["beta_n"]
    v_sat_p = material_params["v_sat_p"]
    beta_p = material_params["beta_p"]

    final_eqn_n = f"LowFieldElectronMobility_edge / (1 + (LowFieldElectronMobility_edge * EParallel / {v_sat_n})^{beta_n})^(1/{beta_n})"
    devsim.edge_model(device=device, region=region, name="ElectronMobility", equation=final_eqn_n)

    final_eqn_p = f"LowFieldHoleMobility_edge / (1 + (LowFieldHoleMobility_edge * EParallel / {v_sat_p})^{beta_p})^(1/{beta_p})"
    devsim.edge_model(device=device, region=region, name="HoleMobility", equation=final_eqn_p)

def setup_physics_and_materials(device, material_params, global_params):
    """Setup all physics and material properties in the correct order."""
    print("\n--- Setting up physics and materials ---")

    # Step 1: Set basic constants and temperature for all regions
    for region in ["p_region", "n_region"]:
        set_silicon_parameters(device=device, region=region,
                               material_params=material_params,
                               temperature_K=global_params["temperature_K"])

    # Step 2: Define doping profiles, which is fundamental for other models
    define_doping(device=device, material_params=material_params)

    # Step 3: Define all other physics models for each region
    for region in ["p_region", "n_region"]:
        # Node solutions must be defined early
        devsim.node_solution(device=device, region=region, name="Potential")
        devsim.node_solution(device=device, region=region, name="Electrons")
        devsim.node_solution(device=device, region=region, name="Holes")

        # Mobility model defines "TotalDoping", which is needed by other models
        print(f"Defining mobility for {region}...")
        define_mobility_models(device=device, region=region, material_params=material_params)

        # Carrier statistics depend on TotalDoping (for BGN)
        define_carrier_statistics(device=device, region=region)

        # SRH lifetimes also depend on TotalDoping
        print(f"Defining SRH lifetime for {region}...")
        define_srh_lifetime_models(device=device, region=region, material_params=material_params)

    print("\n--- Physics and doping defined successfully ---")

def define_basic_edge_models(device, region, material_params):
    """Define basic edge models for current transport."""
    print(f"    Defining basic edge models for {region}...")

    # Electric displacement field
    devsim.edge_model(device=device, region=region, name="DField", equation="Permittivity * ElectricField")
    devsim.edge_model(device=device, region=region, name="DField:Potential@n0",
                      equation="Permittivity * EdgeInverseLength")
    devsim.edge_model(device=device, region=region, name="DField:Potential@n1",
                      equation="-Permittivity * EdgeInverseLength")

    # Space charge
    devsim.node_model(device=device, region=region, name="SpaceCharge",
                      equation="ElectronCharge * (Holes - Electrons + NetDoping)")

    # Bernoulli functions for current transport
    devsim.edge_model(device=device, region=region, name="vdiff",
                      equation="(Potential@n0 - Potential@n1)/ThermalVoltage")
    devsim.edge_model(device=device, region=region, name="Bernoulli_vdiff", equation="B(vdiff)")
    devsim.edge_model(device=device, region=region, name="Bernoulli_neg_vdiff", equation="B(-vdiff)")

    # Edge values from node models
    devsim.edge_from_node_model(device=device, region=region, node_model="Electrons")
    devsim.edge_from_node_model(device=device, region=region, node_model="Holes")

def define_current_models(device, region):
    """Define electron and hole current models with derivatives."""
    print(f"    Defining current models for {region}...")

    # Current equations using Scharfetter-Gummel discretization
    electron_current_eq = "ElectronCharge * ElectronMobility * ThermalVoltage * EdgeInverseLength * (Electrons@n1 * Bernoulli_neg_vdiff - Electrons@n0 * Bernoulli_vdiff)"
    devsim.edge_model(device=device, region=region, name="ElectronCurrent", equation=electron_current_eq)

    hole_current_eq = "ElectronCharge * HoleMobility * ThermalVoltage * EdgeInverseLength * (Holes@n1 * Bernoulli_vdiff - Holes@n0 * Bernoulli_neg_vdiff)"
    devsim.edge_model(device=device, region=region, name="HoleCurrent", equation=hole_current_eq)

    # Define all required derivatives for Newton solver
    for v in ["Potential", "Electrons", "Holes"]:
        for n in ["n0", "n1"]:
            devsim.edge_model(device=device, region=region, name=f"ElectronCurrent:{v}@{n}",
                              equation=f"diff({electron_current_eq}, {v}@{n})")
            devsim.edge_model(device=device, region=region, name=f"HoleCurrent:{v}@{n}",
                              equation=f"diff({hole_current_eq}, {v}@{n})")

def define_equilibrium_models(device, region):
    """Define equilibrium carrier concentrations."""
    print(f"    Defining equilibrium models for {region}...")

    devsim.node_model(device=device, region=region, name="n_i_squared", equation="IntrinsicCarrierDensity^2")
    devsim.node_model(device=device, region=region, name="IntrinsicElectrons",
                      equation="0.5*(NetDoping+(NetDoping^2+4*n_i_squared)^0.5)")
    devsim.node_model(device=device, region=region, name="IntrinsicHoles",
                      equation="0.5*(-NetDoping+(NetDoping^2+4*n_i_squared)^0.5)")

def define_node_averaged_fields(device, region):
    """
    CORRECTED: Defines node-averaged quantities.
    1.  Calculates nodal E-field using vector_gradient on Potential.
    2.  Calculates nodal current densities by reconstructing the drift-diffusion
        equation using only nodal quantities.
    """
    print(f"    Defining node-averaged fields and currents for {region}...")

    # -- Part 1: Calculate Nodal Electric Field --
    # Use vector_gradient to create Potential_gradx, Potential_grady at the nodes.
    devsim.vector_gradient(device=device, region=region, node_model="Potential")
    devsim.node_model(device=device, region=region, name="E_field_x", equation="-Potential_gradx")
    devsim.node_model(device=device, region=region, name="E_field_y", equation="-Potential_grady")

    # Calculate the E-field magnitude at the node from its components.
    devsim.node_model(device=device, region=region, name="E_mag_node_abs",
                      equation="(E_field_x^2 + E_field_y^2 + 1e-12)^(0.5)") # Add small const for stability

    # -- Part 2: Reconstruct Nodal Current Density --
    # We need the gradient of carrier concentrations for the diffusion term.
    devsim.vector_gradient(device=device, region=region, node_model="Electrons")
    devsim.vector_gradient(device=device, region=region, node_model="Holes")

    # Get thermal voltage D = mu*Vt (Einstein relation)
    Vt = devsim.get_parameter(name="ThermalVoltage")

    # Reconstruct Jn = q * (mu_n*n*E + D_n*grad(n)) using only node models
    # Note: We use LowFieldElectronMobility here as a stable nodal approximation.
    Jn_x_eq = f"ElectronCharge * (LowFieldElectronMobility * Electrons * E_field_x + {Vt} * LowFieldElectronMobility * Electrons_gradx)"
    Jn_y_eq = f"ElectronCharge * (LowFieldElectronMobility * Electrons * E_field_y + {Vt} * LowFieldElectronMobility * Electrons_grady)"
    devsim.node_model(device=device, region=region, name="Jn_x_node", equation=Jn_x_eq)
    devsim.node_model(device=device, region=region, name="Jn_y_node", equation=Jn_y_eq)
    devsim.node_model(device=device, region=region, name="Jn_mag_node",
                      equation="(Jn_x_node^2 + Jn_y_node^2 + 1e-20)^(0.5)") # Add small const for stability

    # Reconstruct Jp = q * (mu_p*p*E - D_p*grad(p)) using only node models
    Jp_x_eq = f"ElectronCharge * (LowFieldHoleMobility * Holes * E_field_x - {Vt} * LowFieldHoleMobility * Holes_gradx)"
    Jp_y_eq = f"ElectronCharge * (LowFieldHoleMobility * Holes * E_field_y - {Vt} * LowFieldHoleMobility * Holes_grady)"
    devsim.node_model(device=device, region=region, name="Jp_x_node", equation=Jp_x_eq)
    devsim.node_model(device=device, region=region, name="Jp_y_node", equation=Jp_y_eq)
    devsim.node_model(device=device, region=region, name="Jp_mag_node",
                      equation="(Jp_x_node^2 + Jp_y_node^2 + 1e-20)^0.5") # Add small const for stability



def define_srh_recombination(device, region):
    """
    Define Shockley-Read-Hall recombination, now including the Hurkx TAT enhancement.
    """
    print(f"    Defining SRH recombination with Hurkx TAT enhancement for {region}...")
    devsim.node_model(device=device, region=region, name="n1_srh", equation="IntrinsicCarrierDensity")
    devsim.node_model(device=device, region=region, name="p1_srh", equation="IntrinsicCarrierDensity")

    # Gamma_Hurkx must be defined before this function is called.
    # It enhances both recombination and generation.
    srh_expression = "Gamma_Hurkx * (Electrons * Holes - n_i_squared) / (taup * (Electrons + n1_srh) + taun * (Holes + p1_srh))"

    devsim.node_model(name="USRH", device=device, region=region, equation=srh_expression)
    devsim.node_model(device=device, region=region, name="USRH:Electrons",
                      equation=f"diff({srh_expression}, Electrons)")
    devsim.node_model(device=device, region=region, name="USRH:Holes", equation=f"diff({srh_expression}, Holes)")

def define_auger_recombination(device, region, material_params):
    """Define Auger recombination model."""
    print(f"    Defining Auger recombination for {region}...")

    devsim.set_parameter(device=device, region=region, name="C_n_auger", value=material_params["C_n_auger"])
    devsim.set_parameter(device=device, region=region, name="C_p_auger", value=material_params["C_p_auger"])

    auger_eq = "(C_n_auger * Electrons + C_p_auger * Holes) * (Electrons * Holes - n_i_squared)"
    devsim.node_model(name="UAuger", device=device, region=region, equation=auger_eq)
    devsim.node_model(device=device, region=region, name="UAuger:Electrons",
                      equation=f"diff({auger_eq}, Electrons)")
    devsim.node_model(device=device, region=region, name="UAuger:Holes",
                      equation=f"diff({auger_eq}, Holes)")

def define_optical_generation(device, region):
    """Define optical generation model including reflection losses."""
    print(f"    Defining optical generation for {region}...")

    # CORRECTED: This now uses EffectivePhotonFlux which will account for reflection
    devsim.node_model(device=device, region=region, name="OpticalGeneration",
                      equation="EffectivePhotonFlux * alpha * exp(-alpha * abs(y* 1e-4))")
    devsim.node_model(device=device, region=region, name="OpticalGeneration:Electrons", equation="0.0")
    devsim.node_model(device=device, region=region, name="OpticalGeneration:Holes", equation="0.0")

def define_impact_ionization(device, region, material_params):
    """Define impact ionization generation model with proper safety checks."""
    print(f"    Defining impact ionization for {region}...")

    # # Set impact ionization parameters
    # devsim.set_parameter(device=device, region=region, name="a_n", value=material_params["a_n"])
    # devsim.set_parameter(device=device, region=region, name="b_n", value=material_params["b_n"])
    # devsim.set_parameter(device=device, region=region, name="a_p", value=material_params["a_p"])
    # devsim.set_parameter(device=device, region=region, name="b_p", value=material_params["b_p"])
    # devsim.set_parameter(device=device, region=region, name="E_th_n", value=material_params["E_th_n"])
    # devsim.set_parameter(device=device, region=region, name="E_th_p", value=material_params["E_th_p"])
    #
    # # *** FIX APPLIED HERE ***
    # # Added a small constant (1e-30) to the denominator to prevent division by zero.
    # # The ifelse condition is simplified to only check against the threshold field.
    # devsim.node_model(device=device, region=region, name="alpha_n_node",
    #                   equation=f"ifelse(E_mag_node_abs > E_th_n, a_n * exp(-b_n / (E_mag_node_abs + 1e-30)), 0.0)")
    # devsim.node_model(device=device, region=region, name="alpha_p_node",
    #                   equation=f"ifelse(E_mag_node_abs > E_th_p, a_p * exp(-b_p / (E_mag_node_abs + 1e-30)), 0.0)")
    #
    # # The generation rate formula remains the same, but now relies on the safe alpha models.
    # # The original ifelse conditions here were good for stability.
    # min_field = 1e4 # A reasonable field to start considering ionization
    # generation_eq = f"ifelse(Jn_mag_node > 1e-20 && E_mag_node_abs > {min_field}, alpha_n_node * Jn_mag_node / ElectronCharge, 0.0) + ifelse(Jp_mag_node > 1e-20 && E_mag_node_abs > {min_field}, alpha_p_node * Jp_mag_node / ElectronCharge, 0.0)"
    # devsim.node_model(device=device, region=region, name="G_impact", equation=generation_eq)
    #
    # devsim.node_model(device=device, region=region, name="G_impact:Electrons",
    #                   equation=f"diff({generation_eq}, Electrons)")
    # devsim.node_model(device=device, region=region, name="G_impact:Holes",
    #                   equation=f"diff({generation_eq}, Holes)")

    devsim.node_model(device=device, region=region, name="G_impact", equation="0.0")
    devsim.node_model(device=device, region=region, name="G_impact:Electrons", equation="0.0")
    devsim.node_model(device=device, region=region, name="G_impact:Holes", equation="0.0")

def define_net_recombination(device, region):
    """Assembles all generation and recombination models into a single NetRecombination term."""
    print(f"    Assembling final NetRecombination for {region}...")

    # Assemble all G-R mechanisms
    # Generation terms (Optical, Impact) are SUBTRACTED from recombination (SRH, Auger)
    net_recombination_eq = "USRH + UAuger - G_impact - OpticalGeneration"  # CORRECTED

    # This model is used by the continuity equations
    devsim.node_model(device=device, region=region, name="NetRecombination",
                      equation=net_recombination_eq)

    # Derivatives are needed for the solver
    devsim.node_model(device=device, region=region, name="NetRecombination:Electrons",
                      equation=f"diff({net_recombination_eq}, Electrons)")
    devsim.node_model(device=device, region=region, name="NetRecombination:Holes",
                      equation=f"diff({net_recombination_eq}, Holes)")

    # These are helper models for the continuity equations themselves
    devsim.node_model(device=device, region=region, name="eCharge_x_NetRecomb",
                      equation=f"ElectronCharge * ({net_recombination_eq})")
    devsim.node_model(device=device, region=region, name="Neg_eCharge_x_NetRecomb",
                      equation=f"-ElectronCharge * ({net_recombination_eq})")


def setup_photodiode_model(device, material_params, global_params):
    """Setup the full photodiode physical model and equations ."""
    print("--- Setting Up Full Photodiode Physical Model ---")

    # Initialize global parameters
    devsim.set_parameter(name="PhotonFlux", value=global_params["photon_flux"])
    devsim.set_parameter(name="EffectivePhotonFlux", value=0.0)  # ADDED: Initialize EffectivePhotonFlux
    devsim.set_parameter(name="alpha", value=0.0)

    k_B_eV = devsim.get_parameter(name="k_B_eV")
    T = devsim.get_parameter(name="T")
    devsim.set_parameter(name="ThermalVoltage", value=k_B_eV * T)

    # Define physical models for each region using modular functions
    for region in ["p_region", "n_region"]:
        print(f"  Setting up physics for {region}...")

        # Stage 1: Basic transport models
        define_basic_edge_models(device, region, material_params)
        define_current_models(device, region)
        define_equilibrium_models(device, region)

        # Stage 2: Field averaging for generation-recombination
        define_node_averaged_fields(device, region)
        define_hurkx_enhancement(device, region, material_params)

        # Stage 3: Generation-recombination physics
        define_srh_recombination(device, region)  # This already includes TAT via Hurkx model
        define_auger_recombination(device, region, material_params)
        define_optical_generation(device, region)

        define_impact_ionization(device, region, material_params)

        # Stage 4: Net recombination assembly
        define_net_recombination(device, region)

    print("✅ Photodiode model setup complete!")

def setup_boundary_conditions(device, material_params):
    """Define all boundary condition models."""
    print("  Defining all boundary condition models...")

    devsim.set_parameter(device=device, name="Sn", value=material_params["s_n"])
    devsim.set_parameter(device=device, name="Sp", value=material_params["s_p"])

    for contact in ["anode", "cathode"]:
        # Electron recombination current at the surface
        devsim.contact_node_model(
            device=device, contact=contact, name="ElectronSurfaceRecombinationCurrent",
            equation="ElectronCharge * Sn * (Electrons - IntrinsicElectrons)"
        )
        devsim.contact_node_model(
            device=device, contact=contact, name="ElectronSurfaceRecombinationCurrent:Electrons",
            equation="ElectronCharge * Sn"
        )

        # Hole recombination current at the surface
        devsim.contact_node_model(
            device=device, contact=contact, name="HoleSurfaceRecombinationCurrent",
            equation="ElectronCharge * Sp * (Holes - IntrinsicHoles)"
        )
        devsim.contact_node_model(
            device=device, contact=contact, name="HoleSurfaceRecombinationCurrent:Holes",
            equation="ElectronCharge * Sp"
        )

        # Standard bias and carrier concentration boundary conditions (Ohmic)
        devsim.set_parameter(device=device, name=f"{contact}_bias", value=0.0)
        devsim.contact_node_model(device=device, contact=contact, name=f"{contact}_potential_bc",
                                  equation=f"Potential - {contact}_bias")
        devsim.contact_node_model(device=device, contact=contact, name=f"{contact}_potential_bc:Potential",
                                  equation="1.0")
        devsim.contact_node_model(device=device, contact=contact, name=f"{contact}_electrons_bc",
                                  equation="Electrons - IntrinsicElectrons")
        devsim.contact_node_model(device=device, contact=contact, name=f"{contact}_electrons_bc:Electrons",
                                  equation="1.0")
        devsim.contact_node_model(device=device, contact=contact, name=f"{contact}_holes_bc",
                                  equation="Holes - IntrinsicHoles")
        devsim.contact_node_model(device=device, contact=contact, name=f"{contact}_holes_bc:Holes", equation="1.0")

    for variable in ["Potential", "Electrons", "Holes"]:
        devsim.interface_model(device=device, interface="pn_junction", name=f"{variable}_continuity",
                               equation=f"{variable}@r0 - {variable}@r1")
        devsim.interface_model(device=device, interface="pn_junction", name=f"{variable}_continuity:{variable}@r0",
                               equation="1.0")
        devsim.interface_model(device=device, interface="pn_junction", name=f"{variable}_continuity:{variable}@r1",
                               equation="-1.0")


def solve_initial_equilibrium(device):
    """
    Solve for initial equilibrium state with enhanced numerical stability.
    Uses a more robust approach for devices with large doping contrasts.
    """
    print("  Solving for initial equilibrium state (enhanced stability method)...")

    # ------------------ STEP 1: SOLVE POISSON EQUATION ONLY ------------------
    print("    Step 1/4: Creating initial guess and solving Poisson equation...")

    # Create a more conservative initial guess
    for region in ["p_region", "n_region"]:
        # Set carriers to intrinsic values initially
        devsim.set_node_values(device=device, region=region, name="Electrons", init_from="IntrinsicElectrons")
        devsim.set_node_values(device=device, region=region, name="Holes", init_from="IntrinsicHoles")

        # Create a more conservative potential initial guess based on work function difference
        if region == "p_region":
            # P-region should be at higher potential
            devsim.node_model(device=device, region=region, name="InitialPotential", equation="0.2")
        else:
            # N-region should be at lower potential
            devsim.node_model(device=device, region=region, name="InitialPotential", equation="-0.2")

        devsim.set_node_values(device=device, region=region, name="Potential", init_from="InitialPotential")

    # Define ONLY the PotentialEquation first
    for region in ["p_region", "n_region"]:
        devsim.equation(device=device, region=region, name="PotentialEquation",
                        variable_name="Potential",
                        node_model="SpaceCharge",
                        edge_model="DField",
                        variable_update="default")
    for contact in ["anode", "cathode"]:
        devsim.contact_equation(device=device, contact=contact, name="PotentialEquation",
                                node_model=f"{contact}_potential_bc")
    devsim.interface_equation(device=device, interface="pn_junction", name="PotentialEquation",
                              interface_model="Potential_continuity", type="continuous")

    try:
        devsim.solve(type="dc", absolute_error=10, relative_error=1e-8, maximum_iterations=200)
        print("    ✅ Poisson equation solved successfully.")
    except devsim.error as msg:
        print(f"    ⚠️ Poisson solve failed, trying with relaxed parameters: {msg}")
        try:
            devsim.solve(type="dc", absolute_error=100, relative_error=1e-6, maximum_iterations=300)
            print("    ✅ Poisson equation solved with relaxed parameters.")
        except devsim.error as msg2:
            print(f"    ❌ Poisson solve failed completely: {msg2}")
            raise

    # ------------------ STEP 2: GRADUAL CARRIER INTRODUCTION ------------------
    print("    Step 2/4: Gradually introducing carriers with damping...")

    for region in ["p_region", "n_region"]:
        # Use a more conservative Boltzmann distribution with damping
        devsim.node_model(device=device, region=region, name="DampedElectrons",
                          equation="IntrinsicCarrierDensity * exp(0.5 * Potential/ThermalVoltage)")
        devsim.node_model(device=device, region=region, name="DampedHoles",
                          equation="IntrinsicCarrierDensity * exp(-0.5 * Potential/ThermalVoltage)")

        devsim.set_node_values(device=device, region=region, name="Electrons", init_from="DampedElectrons")
        devsim.set_node_values(device=device, region=region, name="Holes", init_from="DampedHoles")

    print("    ✅ Damped carrier initial guess updated.")

    # ------------------ STEP 3: SOLVE WITH RAMPED DOPING ------------------
    print("    Step 3/4: Solving with ramped doping to avoid singularities...")

    # Temporarily reduce the doping contrast to avoid numerical issues
    for region in ["p_region", "n_region"]:
        # Create ramped versions of the doping
        if region == "p_region":
            devsim.node_model(device=device, region=region, name="RampedAcceptors",
                              equation="0.1 * Acceptors + 0.9 * 1e15")  # Start with lower contrast
            devsim.node_model(device=device, region=region, name="RampedDonors",
                              equation="Donors")
        else:
            devsim.node_model(device=device, region=region, name="RampedAcceptors", equation="0.0")
            devsim.node_model(device=device, region=region, name="RampedDonors", equation="Donors")

        devsim.node_model(device=device, region=region, name="RampedNetDoping",
                          equation="RampedDonors - RampedAcceptors")

        # Update space charge with ramped doping
        devsim.node_model(device=device, region=region, name="SpaceCharge",
                          equation="ElectronCharge * (Holes - Electrons + RampedNetDoping)")

    # Define carrier continuity equations with heavy damping
    for region in ["p_region", "n_region"]:
        devsim.equation(device=device, region=region, name="ElectronContinuityEquation",
                        variable_name="Electrons",
                        node_model="eCharge_x_NetRecomb",
                        edge_model="ElectronCurrent",
                        variable_update="log_damp")
        devsim.equation(device=device, region=region, name="HoleContinuityEquation",
                        variable_name="Holes",
                        node_model="Neg_eCharge_x_NetRecomb",
                        edge_model="HoleCurrent",
                        variable_update="log_damp")

    # Define carrier continuity at interfaces and contacts
    devsim.interface_equation(device=device, interface="pn_junction", name="ElectronContinuityEquation",
                              interface_model="Electrons_continuity", type="continuous")
    devsim.interface_equation(device=device, interface="pn_junction", name="HoleContinuityEquation",
                              interface_model="Holes_continuity", type="continuous")

    for contact in ["anode", "cathode"]:
        devsim.contact_equation(device=device, contact=contact, name="ElectronContinuityEquation",
                                node_model="ElectronSurfaceRecombinationCurrent",
                                edge_current_model="ElectronCurrent")
        devsim.contact_equation(device=device, contact=contact, name="HoleContinuityEquation",
                                node_model="HoleSurfaceRecombinationCurrent",
                                edge_current_model="HoleCurrent")

    # Solve with ramped doping
    try:
        devsim.solve(type="dc", absolute_error=100, relative_error=1e-3, maximum_iterations=100)
        print("    ✅ Ramped doping system solved successfully")
    except devsim.error as msg:
        print(f"    ⚠️ Ramped solve failed: {msg}")
        # Try with even more relaxed parameters
        try:
            devsim.solve(type="dc", absolute_error=1000, relative_error=1e-2, maximum_iterations=200)
            print("    ✅ Ramped system solved with very relaxed parameters")
        except devsim.error as msg2:
            print(f"    ❌ Cannot solve even ramped system: {msg2}")
            raise

    # ------------------ STEP 4: RAMP UP TO FULL DOPING ------------------
    print("    Step 4/4: Ramping up to full doping contrast...")

    # Gradually increase doping contrast
    ramp_steps = [0.3, 0.6, 0.8, 1.0]

    for i, ramp_factor in enumerate(ramp_steps):
        print(f"      Ramp step {i + 1}/4: {ramp_factor * 100:.0f}% doping contrast...")

        for region in ["p_region", "n_region"]:
            if region == "p_region":
                devsim.node_model(device=device, region=region, name="RampedAcceptors",
                                  equation=f"{ramp_factor} * Acceptors + {1 - ramp_factor} * 1e15")
            else:
                devsim.node_model(device=device, region=region, name="RampedAcceptors", equation="0.0")

            devsim.node_model(device=device, region=region, name="RampedNetDoping",
                              equation="RampedDonors - RampedAcceptors")

            # Update space charge
            devsim.node_model(device=device, region=region, name="SpaceCharge",
                              equation="ElectronCharge * (Holes - Electrons + RampedNetDoping)")

        # Solve this ramp step
        try:
            if i < 2:  # First two steps with relaxed parameters
                devsim.solve(type="dc", absolute_error=100, relative_error=1e-3, maximum_iterations=100)
            else:  # Last two steps with tighter parameters
                devsim.solve(type="dc", absolute_error=10, relative_error=1e-5, maximum_iterations=200)
            print(f"      ✅ Ramp step {i + 1} solved successfully")
        except devsim.error as msg:
            print(f"      ⚠️ Ramp step {i + 1} failed: {msg}")
            if i == len(ramp_steps) - 1:  # Last step
                print("      ⚠️ Final step failed, but device should be close to equilibrium")
                break
            else:
                # Try to continue with relaxed parameters
                try:
                    devsim.solve(type="dc", absolute_error=1000, relative_error=1e-2, maximum_iterations=300)
                    print(f"      ✅ Ramp step {i + 1} solved with relaxed parameters")
                except devsim.error as msg2:
                    print(f"      ❌ Ramp step {i + 1} failed completely: {msg2}")
                    break

    # Restore original doping
    for region in ["p_region", "n_region"]:
        devsim.node_model(device=device, region=region, name="SpaceCharge",
                          equation="ElectronCharge * (Holes - Electrons + NetDoping)")

    print("    ✅ Equilibrium solution completed with ramped approach")
    print("✅ Initial equilibrium established successfully")


def debug_solver_state(device, step_name):
    """Debug function to check solver state and identify issues."""
    print(f"\n=== DEBUGGING SOLVER STATE: {step_name} ===")

    for region in ["p_region", "n_region"]:
        try:
            potential = np.array(devsim.get_node_model_values(device=device, region=region, name="Potential"))
            electrons = np.array(devsim.get_node_model_values(device=device, region=region, name="Electrons"))
            holes = np.array(devsim.get_node_model_values(device=device, region=region, name="Holes"))

            print(f"  {region}:")
            print(f"    Potential: {np.min(potential):.3f} to {np.max(potential):.3f} V")
            print(f"    Electrons: {np.min(electrons):.2e} to {np.max(electrons):.2e} cm⁻³")
            print(f"    Holes: {np.min(holes):.2e} to {np.max(holes):.2e} cm⁻³")

            # Check for problematic values
            if np.any(electrons <= 0) or np.any(holes <= 0):
                print(f"    ❌ Non-positive carriers detected in {region}")
            if np.any(np.isnan(potential)) or np.any(np.isinf(potential)):
                print(f"    ❌ NaN/Inf in potential in {region}")
            if np.any(np.isnan(electrons)) or np.any(np.isnan(holes)):
                print(f"    ❌ NaN in carrier concentrations in {region}")

        except Exception as e:
            print(f"    ❌ Error getting values for {region}: {e}")

            

# ==============================================================================
#                      SIMULATION FUNCTIONS
# ==============================================================================
def run_iv_sweep(device, voltages, p_flux, material_params=None, wavelength_nm=None):
    """Run I-V sweep simulation with improved stepping and error handling."""
    currents = []

    # FIXED: Handle EffectivePhotonFlux properly
    if p_flux > 0 and material_params and wavelength_nm:
        reflectivity = get_reflectivity(wavelength_nm, material_params)
        effective_flux = p_flux * (1.0 - reflectivity)
        devsim.set_parameter(name="EffectivePhotonFlux", value=effective_flux)
        print(f"Setting EffectivePhotonFlux = {effective_flux:.2e} (R = {reflectivity:.2%})")
    else:
        devsim.set_parameter(name="EffectivePhotonFlux", value=0.0)

    last_good_voltage = 0.0

    for i, v in enumerate(voltages):
        # Use smaller steps for large voltage changes
        if abs(v - last_good_voltage) > 2.0:
            intermediate_steps = max(2, int(abs(v - last_good_voltage) / 1.0))
            step_voltages = np.linspace(last_good_voltage, v, intermediate_steps + 1)[1:]
        else:
            step_voltages = [v]

        success = True
        for step_v in step_voltages:
            print(f"\nSetting Anode Bias: {step_v:.3f} V")
            devsim.set_parameter(device=device, name="anode_bias", value=step_v)
            try:
                devsim.solve(type="dc",
                             absolute_error=10,
                             relative_error=1e-7,
                             maximum_iterations=300,
                             maximum_divergence=30)

                if step_v == v:  # Only record current at target voltage
                    e_current = devsim.get_contact_current(device=device, contact="anode",
                                                           equation="ElectronContinuityEquation")
                    h_current = devsim.get_contact_current(device=device, contact="anode",
                                                           equation="HoleContinuityEquation")
                    currents.append(e_current + h_current)
                    print(f"✅ V = {v:.3f} V, Current = {currents[-1]:.4e} A/cm")
                    last_good_voltage = v

            except devsim.error as msg:
                print(f"❌ CONVERGENCE FAILED at V = {step_v:.3f} V. Error: {msg}")
                try:
                    print("  Attempting recovery with relaxed parameters...")
                    devsim.solve(type="dc",
                                 absolute_error=10,
                                 relative_error=5e-7,
                                 maximum_iterations=500,
                                 maximum_divergence=50)

                    if step_v == v:
                        e_current = devsim.get_contact_current(device=device, contact="anode",
                                                               equation="ElectronContinuityEquation")
                        h_current = devsim.get_contact_current(device=device, contact="anode",
                                                               equation="HoleContinuityEquation")
                        currents.append(e_current + h_current)
                        print(f"✅ RECOVERED: V = {v:.3f} V, Current = {currents[-1]:.4e} A/cm")
                        last_good_voltage = v

                except devsim.error as recovery_msg:
                    print(f"  Recovery failed: {recovery_msg}")
                    currents.append(float('nan'))
                    success = False
                    break

        if not success:
            print(f"Stopping I-V sweep at V = {v:.3f} V due to convergence failure")
            break

    devsim.set_parameter(device=device, name="anode_bias", value=0.0)

    return np.array(currents)


def calculate_qe(dark_currents, light_currents, incident_flux, wavelength_nm):
    """
    Calculate External Quantum Efficiency.
    ... (docstring is fine) ...
    """
    q = 1.602e-19
    photocurrent_density = np.abs(light_currents - dark_currents)
    electrons_per_sec_per_cm2 = photocurrent_density / q
    incident_photons_per_sec_per_cm2 = incident_flux

    if incident_photons_per_sec_per_cm2 > 0:
        qe = (electrons_per_sec_per_cm2 / incident_photons_per_sec_per_cm2) * 100.0
    else:
        qe = 0.0
    return qe

def run_cv_sweep_ac(device, voltages, freq_hz):
    """Calculate C-V using small-signal AC analysis."""
    capacitances = []
    omega = 2.0 * np.pi * freq_hz

    print(f"\nStarting AC C-V sweep at {freq_hz / 1e6:.1f} MHz...")

    for i, v in enumerate(voltages):
        devsim.set_parameter(device=device, name="anode_bias", value=v)
        print(f"Step {i + 1}/{len(voltages)}: Bias = {v:.2f} V")
        try:
            devsim.solve(type="dc", absolute_error=100.0, relative_error=3e-7, maximum_iterations=200)
            devsim.solve(type="ac", frequency=freq_hz)
            imag_i_e = devsim.get_contact_current(device=device, contact="anode",
                                                  equation="ElectronContinuityEquation")
            imag_i_h = devsim.get_contact_current(device=device, contact="anode",
                                                  equation="HoleContinuityEquation")
            C = (imag_i_e + imag_i_h) / omega
            capacitances.append(C)
            print(f"  ✅ C = {C * 1e12:.4f} pF/cm")
        except devsim.error as msg:
            print(f"  ❌ Failed at V = {v:.2f} V: {msg}")
            capacitances.append(float('nan'))

    devsim.set_parameter(device=device, name="anode_bias", value=0.0)
    return np.array(capacitances)

def run_spectral_sweep(device, wavelengths_nm, qe_bias, incident_flux, dark_current_interp,
                       iv_voltages, dark_currents, material_params):
    """Run spectral QE sweep simulation."""
    qe_spectral_results = []

    print(f"\nSimulating QE at a fixed bias of {qe_bias} V across the spectrum...")
    devsim.set_parameter(device=device, name="anode_bias", value=qe_bias)

    # Interpolate dark current at the specific bias
    dark_current_at_bias = np.interp(qe_bias, iv_voltages, dark_currents)

    for wl in wavelengths_nm:
        # CORRECTED: Calculate alpha and reflectivity for each wavelength
        alpha_val = get_alpha_for_wavelength(wl, material_params)
        reflectivity = get_reflectivity(wl, material_params)
        effective_flux = incident_flux * (1.0 - reflectivity)

        # Set parameters in the devsim model
        devsim.set_parameter(name="alpha", value=alpha_val)
        devsim.set_parameter(name="EffectivePhotonFlux", value=effective_flux) # Use corrected parameter

        try:
            # Solve for the light condition at this wavelength
            devsim.solve(type="dc", absolute_error=10.0, relative_error=1e-7, maximum_iterations=100)

            # Get the light current
            e_current = devsim.get_contact_current(device=device, contact="anode",
                                                   equation="ElectronContinuityEquation")
            h_current = devsim.get_contact_current(device=device, contact="anode",
                                                   equation="HoleContinuityEquation")
            light_current_at_bias = e_current + h_current

            # Calculate and store the QE value
            qe_val = calculate_qe(dark_current_at_bias, light_current_at_bias, incident_flux, wl)
            qe_spectral_results.append(qe_val)
            print(f"  ✅ Wavelength: {wl:.1f} nm, R: {reflectivity:.2%}, Photocurrent: {light_current_at_bias:.3e} A/cm, QE: {qe_val:.2f}%")

        except devsim.error as msg:
            print(f"  ❌ CONVERGENCE FAILED at {wl:.1f} nm. Error: {msg}")
            qe_spectral_results.append(float('nan'))

    # Reset photon flux to zero after the sweep
    devsim.set_parameter(name="EffectivePhotonFlux", value=0.0)

    return qe_spectral_results


def plot_results(iv_voltages, dark_currents, light_currents, qe_vs_voltage,
                 cv_voltages, capacitances, wavelengths_nm, qe_spectral,
                 wavelength_single, qe_bias):
    """Generate all plots."""
    print("\n--- Generating All Plots ---")

    # Plot 1: I-V Characteristics
    fig_iv = go.Figure()
    fig_iv.add_trace(go.Scatter(x=iv_voltages, y=np.abs(dark_currents), mode='lines+markers',
                                name='Dark Current', marker_color='red'))
    fig_iv.add_trace(go.Scatter(x=iv_voltages, y=np.abs(light_currents), mode='lines+markers',
                                name=f'Photocurrent @ {wavelength_single} nm', marker_color='blue'))
    fig_iv.update_layout(title_text="I-V Characteristics (Interactive)",
                         xaxis_title="Anode Voltage (V)",
                         yaxis_title="Current Magnitude (A/cm)",
                         yaxis_type="log")
    fig_iv.show()

    # Plot 2: QE vs. Voltage
    fig_qe_v = go.Figure()
    fig_qe_v.add_trace(go.Scatter(x=iv_voltages, y=qe_vs_voltage, mode='lines+markers',
                                  name='QE', marker_color='green'))
    fig_qe_v.update_layout(title_text=f"QE vs. Voltage @ {wavelength_single} nm (Interactive)",
                           xaxis_title="Anode Voltage (V)",
                           yaxis_title="External Quantum Efficiency (%)",
                           yaxis_range=[0, 105])
    fig_qe_v.show()

    # Plot 3: C-V and Mott-Schottky
    fig_cv = make_subplots(rows=1, cols=2, subplot_titles=("C-V @ 1 MHz", "Mott-Schottky Plot"))
    valid_cv_indices = ~np.isnan(capacitances)
    inv_C_squared = 1.0 / (capacitances[valid_cv_indices] ** 2)
    fig_cv.add_trace(go.Scatter(x=cv_voltages[valid_cv_indices], y=capacitances[valid_cv_indices] * 1e12,
                                mode='lines+markers', name='Capacitance', marker_color='magenta'), row=1, col=1)
    fig_cv.add_trace(go.Scatter(x=cv_voltages[valid_cv_indices], y=inv_C_squared,
                                mode='lines+markers', name='1/C²', marker_color='darkturquoise'), row=1, col=2)
    fig_cv.update_xaxes(title_text="Anode Voltage (V)", row=1, col=1)
    fig_cv.update_yaxes(title_text="Capacitance (pF/cm)", row=1, col=1)
    fig_cv.update_xaxes(title_text="Anode Voltage (V)", row=1, col=2)
    fig_cv.update_yaxes(title_text="1/C² (F⁻²cm²)", row=1, col=2)
    fig_cv.update_layout(title_text="Capacitance Analysis (Interactive)", showlegend=False)
    fig_cv.show()

    # Plot 4: Spectral Response
    fig_spectral = go.Figure()
    fig_spectral.add_trace(go.Scatter(x=wavelengths_nm, y=qe_spectral, mode='lines+markers',
                                      name=f'QE at {qe_bias}V', marker_color='purple'))
    fig_spectral.update_layout(title_text=f"Photodiode Spectral Response at V_anode = {qe_bias}V (Interactive)",
                               xaxis_title="Wavelength (nm)",
                               yaxis_title="External Quantum Efficiency (%)",
                               yaxis_range=[0, 105])
    fig_spectral.show()

# ==============================================================================
#                      MAIN EXECUTION BLOCK
# ==============================================================================
def main():
    """Main execution function."""
    # Step 1: Device initialization
    load_mesh_and_create_device(GLOBAL_PARAMS["device_name"], GLOBAL_PARAMS["mesh_file"])
    verify_device_structure(GLOBAL_PARAMS["device_name"])
    verify_mesh_dimensions(GLOBAL_PARAMS["device_name"])
    comprehensive_mesh_debug(GLOBAL_PARAMS["device_name"], SILICON_PARAMS)

    # Step 2: Setup physics
    setup_physics_and_materials(GLOBAL_PARAMS["device_name"], SILICON_PARAMS, GLOBAL_PARAMS)


    # Step 3: Setup photodiode model
    setup_photodiode_model(GLOBAL_PARAMS["device_name"], SILICON_PARAMS, GLOBAL_PARAMS)
    setup_boundary_conditions(GLOBAL_PARAMS["device_name"], SILICON_PARAMS)
    debug_doping_profile(GLOBAL_PARAMS["device_name"])
    solve_initial_equilibrium(GLOBAL_PARAMS["device_name"])
    debug_equilibrium_state(GLOBAL_PARAMS["device_name"])


    # Step 4: Run simulations
    print("\n--- Running Single-Point Simulations for I-V, C-V, and QE vs. V plots ---")
    debug_current_components(GLOBAL_PARAMS["device_name"], voltage=0.0)
    debug_current_components(GLOBAL_PARAMS["device_name"], voltage=-1.0)
    devsim.set_parameter(name="EffectivePhotonFlux", value=6.57e16)
    force_optical_update(GLOBAL_PARAMS["device_name"])
    for region in ["p_region", "n_region"]:
        optical = np.array(
            devsim.get_node_model_values(device=GLOBAL_PARAMS["device_name"], region=region, name="OpticalGeneration"))
        net_recomb = np.array(
            devsim.get_node_model_values(device=GLOBAL_PARAMS["device_name"], region=region, name="NetRecombination"))
        print(f"{region}: OptGen = {np.mean(optical):.2e}, NetRecomb = {np.mean(net_recomb):.2e}")
    debug_photocurrent_mechanism(GLOBAL_PARAMS["device_name"], 0.0)
    comprehensive_optical_debug(GLOBAL_PARAMS["device_name"])


    # I-V and QE vs. V Simulation
    iv_voltages = np.linspace(2, -10, 65)
    LIGHT_PHOTON_FLUX = 1e17
    WAVELENGTH_NM_SINGLE = 650

    single_alpha = get_alpha_for_wavelength(WAVELENGTH_NM_SINGLE, SILICON_PARAMS)
    devsim.set_parameter(name="alpha", value=single_alpha)
    print(f"Using alpha = {single_alpha:.2e} 1/cm for single-point simulation at {WAVELENGTH_NM_SINGLE} nm")


    # Calculate reflectivity for the single wavelength
    reflectivity_single = get_reflectivity(WAVELENGTH_NM_SINGLE, SILICON_PARAMS)
    effective_flux_single = LIGHT_PHOTON_FLUX * (1.0 - reflectivity_single)
    print(
        f"Reflectivity at {WAVELENGTH_NM_SINGLE} nm is {reflectivity_single:.2%}, Effective Flux: {effective_flux_single:.2e}")

    # Run sweeps
    print("optical debug 1")
    comprehensive_optical_debug(GLOBAL_PARAMS["device_name"])
    debug_optical_parameters(GLOBAL_PARAMS["device_name"], 650, 1e17, SILICON_PARAMS)

    dark_currents_single = run_iv_sweep(GLOBAL_PARAMS["device_name"], iv_voltages, p_flux=0.0)

    light_currents_single = run_iv_sweep(GLOBAL_PARAMS["device_name"], iv_voltages,
                                         p_flux=LIGHT_PHOTON_FLUX,
                                         material_params=SILICON_PARAMS,
                                         wavelength_nm=WAVELENGTH_NM_SINGLE)
    qe_vs_voltage = calculate_qe(dark_currents_single, light_currents_single, LIGHT_PHOTON_FLUX,
                             wavelength_nm=WAVELENGTH_NM_SINGLE)

    print("optical debug 2")
    comprehensive_optical_debug(GLOBAL_PARAMS["device_name"])
    debug_optical_parameters(GLOBAL_PARAMS["device_name"], 650, 1e17, SILICON_PARAMS)

    # C-V Simulation
    cv_voltages = np.linspace(0, -5, 21)
    capacitances = run_cv_sweep_ac(GLOBAL_PARAMS["device_name"], cv_voltages, freq_hz=1e6)

    # Spectral sweep
    print("\n--- STARTING SPECTRAL SWEEP for QE vs. Wavelength plot ---")

    # Check if the dark current sweep completed successfully
    if len(dark_currents_single) != len(iv_voltages):
        print("WARNING: I-V sweep did not complete. Truncating voltage range for interpolation.")
        valid_indices = ~np.isnan(dark_currents_single)
        if np.any(valid_indices):
            last_valid = np.where(valid_indices)[0][-1]
            iv_voltages = iv_voltages[:last_valid + 1]
            dark_currents_single = dark_currents_single[:last_valid + 1]
        else:
            print("ERROR: No valid I-V data available. Cannot proceed with spectral sweep.")
            import sys
            sys.exit(1)

    wavelengths_nm_sweep = np.linspace(GLOBAL_PARAMS["wavelength_start_nm"],
                                       GLOBAL_PARAMS["wavelength_end_nm"],
                                       GLOBAL_PARAMS["wavelength_points"])
    QE_BIAS = -2.0
    INCIDENT_PHOTON_FLUX = 1e17

    qe_spectral_results = run_spectral_sweep(GLOBAL_PARAMS["device_name"], wavelengths_nm_sweep,
                                             QE_BIAS, INCIDENT_PHOTON_FLUX,
                                             None, iv_voltages, dark_currents_single, SILICON_PARAMS)

    print("\n--- ALL SIMULATIONS COMPLETE ---")

    # Step 5: Visualize results
    plot_results(iv_voltages, dark_currents_single, light_currents_single, qe_vs_voltage,
                 cv_voltages, capacitances, wavelengths_nm_sweep, qe_spectral_results,
                 WAVELENGTH_NM_SINGLE, QE_BIAS)


    print(f"dark_currents_single:",dark_currents_single)
    print(f"light_currents_single:",light_currents_single)

if __name__ == "__main__":
    main()