# devsim_simulation_modular.py
#
# Refactored script to support modular material parameters.
# All original functionalities and plots are preserved.

import devsim
import os
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ==============================================================================
#                      1. MATERIAL PARAMETERS DICTIONARY
# ==============================================================================

def get_material_parameters(material_name="silicon"):
    """
    Returns a dictionary containing the physical parameters for the specified material.
    """
    if material_name.lower() == "silicon":
        # All parameters for Silicon are grouped here
        params = {
            # Basic Properties
            "Eg": 1.12,  # Bandgap (eV)
            "epsilon": 11.8,  # Relative permittivity
            "Nc_300K": 2.8e19,  # Conduction band effective density of states at 300K (cm^-3)
            "Nv_300K": 1.04e19,  # Valence band effective density of states at 300K (cm^-3)

            # Caughey-Thomas Mobility Model Parameters for electrons
            "mu_max_n": 1417.0,  # Max mobility (cm^2/V-s)
            "mu_min_n": 68.5,  # Min mobility (cm^2/V-s)
            "N_ref_n": 1.168e17,  # Reference doping (cm^-3)
            "alpha_n": 0.711,  # Exponent

            # Caughey-Thomas Mobility Model Parameters for holes
            "mu_max_p": 470.5,  # Max mobility (cm^2/V-s)
            "mu_min_p": 44.9,  # Min mobility (cm^2/V-s)
            "N_ref_p": 2.23e17,  # Reference doping (cm^-3)
            "alpha_p": 0.719,  # Exponent

            # High-Field Mobility (Velocity Saturation) Parameters
            "vsat_n": 1.07e7,  # Electron saturation velocity (cm/s)
            "beta_n": 2.0,  # Exponent
            "vsat_p": 8.37e6,  # Hole saturation velocity (cm/s)
            "beta_p": 1.0,  # Exponent

            # Klaassen Doping-Dependent SRH Lifetime Model Parameters
            "tau_min_n": 3.52e-7, "tau_max_n": 3.95e-4, "N_SRH_n": 7.1e15,
            "tau_min_p": 3.52e-7, "tau_max_p": 3.95e-4, "N_SRH_p": 7.1e15,

            # Selberherr Impact Ionization Model Parameters
            "a_n": 7.03e5, "b_n": 1.231e6,  # For electrons
            "a_p": 1.582e6, "b_p": 2.036e6,  # For holes

            # Optical constants lookup data (Wavelength in nm, n, k)
            "optical_data": {
                'wl': np.array([400, 450, 500, 550, 600, 650, 700, 800, 900, 1000]),
                'n': np.array([5.57, 4.65, 4.32, 4.07, 3.90, 3.78, 3.69, 3.60, 3.55, 3.52]),
                'k': np.array([0.38, 0.12, 0.06, 0.03, 0.02, 0.015, 0.01, 0.007, 0.004, 0.002])
            }
        }
        return params
    else:
        raise ValueError(f"Material '{material_name}' is not defined.")


# ==============================================================================
#                      2. GLOBAL SIMULATION SETUP
# ==============================================================================
# --- Device and Mesh ---
device = "photodiode"
region = "bulk"
mesh_file = "output/photodiode_mesh.msh"  # Make sure this mesh exists

# --- Doping Parameters ---
peak_p_doping = 5e17
junction_depth = 0.5
doping_straggle = 0.1

# --- Operating Conditions ---
TEMPERATURE_K = 300.0
photon_flux = 0.0

# --- Surface Recombination Velocities (SRV) ---
s_n = 100.0
s_p = 100.0

# --- Select Material ---
material = get_material_parameters("silicon")


# ==============================================================================
#             3. DEVICE SETUP AND PHYSICAL MODEL DEFINITIONS
# ==============================================================================
# (Functions setup_device, define_physical_models, define_doping_profile,
# set_initial_conditions_and_contacts are defined here exactly as in Step 2)
def setup_device(dev, msh_file):
    """Loads the mesh and sets up regions and contacts."""
    if not os.path.exists("output"):
        os.makedirs("output")
    devsim.load_devices(file=msh_file)
    devsim.add_gmsh_region(gmsh_name="bulk", device=dev, region=region)
    devsim.add_gmsh_contact(gmsh_name="anode", device=dev, region=region, material="metal")
    devsim.add_gmsh_contact(gmsh_name="cathode", device=dev, region=region, material="metal")
    devsim.add_gmsh_interface(gmsh_name="top_surface", device=dev, region=region, name="top_surface")


def define_physical_models(dev, reg, mat):
    """
    Defines all the core physical models using parameters from the material dictionary.
    """
    # --- Basic Semiconductor Parameters ---
    devsim.set_parameter(name="T", value=TEMPERATURE_K)
    devsim.set_parameter(name="q", value=1.6e-19)
    devsim.set_parameter(name="k", value=8.617e-5)
    devsim.set_parameter(name="eps_0", value=8.854e-14)
    devsim.set_parameter(name="eps_r", value=mat["epsilon"])
    devsim.set_parameter(name="eps", value=devsim.get_parameter(name="eps_r") * devsim.get_parameter(name="eps_0"))
    devsim.set_parameter(name="V_t", value=devsim.get_parameter(name="k") * devsim.get_parameter(name="T"))

    # --- Intrinsic Carrier Concentration (ni) ---
    devsim.node_model(device=dev, region=reg, name="Nc",
                      equation=f'{mat["Nc_300K"]}*(T/300.0)^1.5')
    devsim.node_model(device=dev, region=reg, name="Nv",
                      equation=f'{mat["Nv_300K"]}*(T/300.0)^1.5')
    devsim.node_model(device=dev, region=reg, name="ni_temp",
                      equation=f'sqrt(Nc*Nv)*exp(-{mat["Eg"]}/(2.0*V_t))')
    devsim.node_model(device=dev, region=reg, name="ni", equation="ni_temp")

    # --- Doping Dependent SRH Lifetime (Klaassen Model) ---
    devsim.node_model(device=dev, region=reg, name="tau_n",
                      equation=f'({mat["tau_max_n"]}) / (1.0 + (N_d+N_a)/{mat["N_SRH_n"]}) + {mat["tau_min_n"]}')
    devsim.node_model(device=dev, region=reg, name="tau_p",
                      equation=f'({mat["tau_max_p"]}) / (1.0 + (N_d+N_a)/{mat["N_SRH_p"]}) + {mat["tau_min_p"]}')

    # --- Caughey-Thomas Doping-Dependent Mobility ---
    devsim.node_model(device=dev, region=reg, name="mu_n_dop",
                      equation=f'{mat["mu_min_n"]} + ({mat["mu_max_n"]} - {mat["mu_min_n"]}) / (1.0 + ((N_d+N_a)/{mat["N_ref_n"]})^{mat["alpha_n"]})')
    devsim.node_model(device=dev, region=reg, name="mu_p_dop",
                      equation=f'{mat["mu_min_p"]} + ({mat["mu_max_p"]} - {mat["mu_min_p"]}) / (1.o + ((N_d+N_a)/{mat["N_ref_p"]})^{mat["alpha_p"]})')

    # --- High-Field (Velocity Saturation) Mobility ---
    devsim.edge_model(device=dev, region=reg, name="E_field_pre", equation="(Potential@n0 - Potential@n1)/EdgeLength")
    devsim.edge_model(device=dev, region=reg, name="E_field", equation="abs(E_field_pre)")
    devsim.edge_model(device=dev, region=reg, name="mu_n_sat",
                      equation=f"2*mu_n_dop/(1 + (1 + (2*mu_n_dop*E_field/{mat["vsat_n"]})^{mat["beta_n"]})^(1/{mat["beta_n"]}))")
    devsim.edge_model(device=dev, region=reg, name="mu_p_sat",
                      equation=f"mu_p_dop/(1 + (mu_p_dop*E_field/{mat["vsat_p"]})^{mat["beta_p"]})^(1/{mat["beta_p"]})")

    # --- Effective Carrier Mobilities ---
    devsim.edge_model(device=dev, region=reg, name="mu_n", equation="mu_n_sat")
    devsim.edge_model(device=dev, region=reg, name="mu_p", equation="mu_p_sat")

    # --- Drift-Diffusion Currents (Scharfetter-Gummel) ---
    devsim.edge_model(device=dev, region=reg, name="vdiff", equation="(Potential@n0 - Potential@n1)/V_t")
    devsim.edge_model(device=dev, region=reg, name="B_n", equation="vdiff/(exp(vdiff)-1)")
    devsim.edge_model(device=dev, region=reg, name="B_p", equation="-vdiff/(exp(-vdiff)-1)")
    devsim.edge_model(device=dev, region=reg, name="Jn",
                      equation="q*mu_n*V_t/EdgeLength * (Electrons@n1*B_n - Electrons@n0*B_p)")
    devsim.edge_model(device=dev, region=reg, name="Jp",
                      equation="-q*mu_p*V_t/EdgeLength * (Holes@n1*B_n - Holes@n0*B_p)")

    # --- SRH Recombination ---
    devsim.node_model(device=dev, region=reg, name="USRH",
                      equation="(Electrons*Holes - ni^2)/(tau_p*(Electrons+ni) + tau_n*(Holes+ni))")

    # --- Optical Generation ---
    devsim.set_parameter(name="photon_flux", value=photon_flux)
    devsim.set_parameter(name="alpha", value=0.0)  # Will be set during sweep
    devsim.node_model(device=dev, region=reg, name="G_opt",
                      equation="photon_flux * alpha * exp(-alpha * y)")

    # --- Total Recombination-Generation ---
    devsim.node_model(device=dev, region=reg, name="R", equation="USRH - G_opt")
    devsim.node_model(device=dev, region=reg, name="NetCharge",
                      equation="q*(Holes - Electrons + N_d - N_a)")

    # --- Impact Ionization (Selberherr Model) ---
    devsim.edge_model(device=dev, region=reg, name="alpha_n_impact",
                      equation=f'{mat["a_n"]}*exp(-{mat["b_n"]}/max(E_field,1e-10))')
    devsim.edge_model(device=dev, region=reg, name="alpha_p_impact",
                      equation=f'{mat["a_p"]}*exp(-{mat["b_p"]}/max(E_field,1e-10))')
    devsim.edge_model(device=dev, region=reg, name="G_impact",
                      equation="0.5/q * (alpha_n_impact*abs(Jn) + alpha_p_impact*abs(Jp))")  # 0.5 factor for edge to node

    # --- Set up Equations ---
    devsim.equation(device=dev, region=reg, name="PotentialEquation",
                    variable_name="Potential", node_model="NetCharge",
                    edge_model="eps", variable_update="default")
    devsim.equation(device=dev, region=reg, name="ElectronContinuityEquation",
                    variable_name="Electrons", time_node_model="q",
                    edge_model="Jn", variable_update="positive",
                    node_model="q*R", edge_charge_model="q*G_impact")
    devsim.equation(device=dev, region=reg, name="HoleContinuityEquation",
                    variable_name="Holes", time_node_model="-q",
                    edge_model="Jp", variable_update="positive",
                    node_model="-q*R", edge_charge_model="-q*G_impact")


def define_doping_profile(dev, reg):
    """Defines the Gaussian and uniform doping profiles."""
    devsim.node_model(device=dev, region=reg, name="N_a",
                      equation=f"{peak_p_doping} * exp(-((y - {junction_depth}) / (sqrt(2) * {doping_straggle}))^2)")
    devsim.node_model(device=dev, region=reg, name="N_d", equation="1e15")
    devsim.node_model(device=dev, region=reg, name="NetDoping", equation="N_d - N_a")


def set_initial_conditions_and_contacts(dev, reg):
    """Sets initial conditions and defines contact equations."""
    devsim.node_solution(device=dev, region=reg, name="Potential")
    devsim.node_solution(device=dev, region=reg, name="Electrons")
    devsim.node_solution(device=dev, region=reg, name="Holes")
    devsim.set_node_values(device=dev, region=reg, name="Potential", init_from="NetDoping",
                           scale="V_t*log(abs(NetDoping)/ni)")
    devsim.set_node_values(device=dev, region=reg, name="Electrons", init_from="ni^2/abs(NetDoping)",
                           where="NetDoping < 0")
    devsim.set_node_values(device=dev, region=reg, name="Electrons", init_from="abs(NetDoping)", where="NetDoping > 0")
    devsim.set_node_values(device=dev, region=reg, name="Holes", init_from="abs(NetDoping)", where="NetDoping < 0")
    devsim.set_node_values(device=dev, region=reg, name="Holes", init_from="ni^2/abs(NetDoping)", where="NetDoping > 0")
    for contact in ["anode", "cathode"]:
        devsim.contact_equation(device=dev, contact=contact, name="PotentialEquation",
                                node_model="NetCharge", edge_model="eps",
                                circuit_node=f"{contact}_bias")
        devsim.contact_equation(device=dev, contact=contact, name="ElectronContinuityEquation",
                                node_model="q*R", edge_model="Jn", edge_charge_model="q*G_impact")
        devsim.contact_equation(device=dev, contact=contact, name="HoleContinuityEquation",
                                node_model="-q*R", edge_model="Jp", edge_charge_model="-q*G_impact")
    devsim.interface_model(device=dev, interface="top_surface", name="U_n_surf",
                           equation=f"{s_n}*(Electrons*Holes - ni^2)/(ni*(Electrons+Holes+2*ni))")
    devsim.interface_model(device=dev, interface="top_surface", name="U_p_surf",
                           equation=f"{s_p}*(Electrons*Holes - ni^2)/(ni*(Electrons+Holes+2*ni))")
    devsim.interface_equation(device=dev, interface="top_surface", name="ElectronContinuityEquation",
                              interface_model="q*U_n_surf", type="flux")
    devsim.interface_equation(device=dev, interface="top_surface", name="HoleContinuityEquation",
                              interface_model="-q*U_p_surf", type="flux")


# ==============================================================================
#                      4. SIMULATION AND ANALYSIS
# ==============================================================================
# (Functions get_optical_constants, run_dark_iv_sweep, run_optical_simulation,
# run_cv_sweep are defined here exactly as in Step 3)
def get_optical_constants(wavelength_nm, mat):
    """Interpolates n and k from the material's optical data table."""
    wl_data = mat['optical_data']['wl']
    n_data = mat['optical_data']['n']
    k_data = mat['optical_data']['k']
    n_r = np.interp(wavelength_nm, wl_data, n_data)
    k_e = np.interp(wavelength_nm, wl_data, k_data)
    return n_r, k_e


def run_dark_iv_sweep(dev):
    """Performs the dark I-V sweep."""
    voltages = np.concatenate([np.linspace(0, 1, 21), np.linspace(1.2, 5, 20), np.linspace(5.5, 20, 30)])
    currents = []
    for v in voltages:
        devsim.set_parameter(name="anode_bias", value=v)
        devsim.solve(type="dc", absolute_error=1e12, relative_error=1e-12, maximum_iterations=100)
        currents.append(devsim.get_contact_current(device=dev, contact="anode", equation="HoleContinuityEquation") +
                        devsim.get_contact_current(device=dev, contact="anode", equation="ElectronContinuityEquation"))
    return voltages, np.array(currents)


def run_optical_simulation(dev, mat):
    """Runs the simulation under illumination to calculate QE."""
    wavelengths_nm_sweep = np.linspace(400, 1000, 61)
    photocurrents = []
    # Set bias for QE calculation (e.g., 0V for short-circuit)
    devsim.set_parameter(name="anode_bias", value=0.0)
    devsim.solve(type="dc", absolute_error=1.0, relative_error=1e-12, maximum_iterations=100)

    for wl_nm in wavelengths_nm_sweep:
        wavelength_m = wl_nm * 1e-9
        n_r, k_e = get_optical_constants(wl_nm, mat)
        alpha_val = 4 * np.pi * k_e / wavelength_m / 100  # Convert to cm^-1

        devsim.set_parameter(name="photon_flux", value=1e17)  # Assume flux of 1e17 photons/cm^2/s
        devsim.set_parameter(name="alpha", value=alpha_val)
        devsim.solve(type="dc", absolute_error=1.0, relative_error=1e-12, maximum_iterations=100)

        photocurrent = (devsim.get_contact_current(device=dev, contact="anode", equation="HoleContinuityEquation") +
                        devsim.get_contact_current(device=dev, contact="anode", equation="ElectronContinuityEquation"))
        photocurrents.append(photocurrent)

    devsim.set_parameter(name="photon_flux", value=0.0)  # Reset flux

    photocurrents = np.array(photocurrents)
    q = devsim.get_parameter("q")
    flux = 1e17  # The flux we used
    qe_spectral = np.abs(photocurrents) / (q * flux) if flux > 0 else np.zeros_like(photocurrents)

    return wavelengths_nm_sweep, qe_spectral


def run_cv_sweep(dev):
    """Performs the C-V sweep."""
    cv_voltages = np.linspace(-5, 0.5, 51)
    capacitances = []
    for v in cv_voltages:
        devsim.set_parameter(name="anode_bias", value=v)
        devsim.solve(type="dc", absolute_error=1e12, relative_error=1e-12, maximum_iterations=100)
        charge = devsim.get_contact_charge(device=dev, contact="anode", equation="PotentialEquation")
        devsim.set_parameter(name="anode_bias", value=v + 0.001)
        devsim.solve(type="dc", absolute_error=1e12, relative_error=1e-12, maximum_iterations=100)
        charge_dv = devsim.get_contact_charge(device=dev, contact="anode", equation="PotentialEquation")
        capacitances.append(abs((charge_dv - charge) / 0.001))
    return cv_voltages, np.array(capacitances)


# ==============================================================================
#                      5. PLOTTING FUNCTIONS
# ==============================================================================
# Your original plotting functions are preserved here without any changes.
def plot_results(voltages, currents, doping_profile, cv_voltages, capacitances, wavelengths_nm_sweep, qe_spectral):
    """Generates all interactive plots."""
    # --- PLOT 1: Doping Profile ---
    fig_doping = go.Figure()
    fig_doping.add_trace(
        go.Scatter(x=doping_profile['y'], y=doping_profile['N_a'], mode='lines', name='Acceptors (N_a)',
                   line=dict(color='red')))
    fig_doping.add_trace(go.Scatter(x=doping_profile['y'], y=doping_profile['N_d'], mode='lines', name='Donors (N_d)',
                                    line=dict(color='blue')))
    fig_doping.add_trace(
        go.Scatter(x=doping_profile['y'], y=doping_profile['NetDoping'], mode='lines', name='Net Doping (N_d - N_a)',
                   line=dict(color='green', dash='dash')))
    fig_doping.update_layout(title='Doping Profile', xaxis_title='Depth (μm)', yaxis_title='Concentration (cm⁻³)',
                             yaxis_type='log', yaxis=dict(range=[14, 18]), xaxis=dict(autorange="reversed"))
    fig_doping.show()

    # --- PLOT 2: Interactive I-V Curve ---
    fig_iv = make_subplots(specs=[[{"secondary_y": True}]])
    fig_iv.add_trace(
        go.Scatter(x=voltages, y=np.abs(currents), name="Dark Current (Linear Scale)", mode='lines+markers',
                   marker_color='blue'), secondary_y=False)
    fig_iv.add_trace(go.Scatter(x=voltages, y=np.abs(currents), name="Dark Current (Log Scale)", mode='lines+markers',
                                marker_color='red'), secondary_y=True)
    fig_iv.update_xaxes(title_text="Anode Voltage (V)")
    fig_iv.update_yaxes(title_text="Current (A/μm)", type="linear", secondary_y=False)
    fig_iv.update_yaxes(title_text="Current (A/μm)", type="log", secondary_y=True)
    fig_iv.update_layout(title_text="Dark I-V Characteristics (Interactive)")
    fig_iv.show()

    # --- PLOT 3: Interactive C-V and 1/C^2 Plot ---
    fig_cv = make_subplots(rows=1, cols=2, subplot_titles=("C-V Curve", "1/C² vs. Voltage"))
    valid_cv_indices = ~np.isnan(capacitances)
    inv_C_squared = 1.0 / (capacitances ** 2)
    fig_cv.add_trace(go.Scatter(x=cv_voltages[valid_cv_indices], y=capacitances[valid_cv_indices] * 1e12,
                                mode='lines+markers', name='Capacitance', marker_color='magenta'), row=1, col=1)
    fig_cv.add_trace(go.Scatter(x=cv_voltages[valid_cv_indices], y=inv_C_squared[valid_cv_indices],
                                mode='lines+markers', name='1/C²', marker_color='darkturquoise'), row=1, col=2)
    fig_cv.update_xaxes(title_text="Anode Voltage (V)", row=1, col=1)
    fig_cv.update_yaxes(title_text="Capacitance (pF/cm)", row=1, col=1)
    fig_cv.update_xaxes(title_text="Anode Voltage (V)", row=1, col=2)
    fig_cv.update_yaxes(title_text="1/C² (F⁻²cm²)", row=1, col=2)
    fig_cv.update_layout(title_text="Capacitance Analysis (Interactive)", showlegend=False)
    fig_cv.show()

    # --- PLOT 4: Interactive Spectral Response Plot ---
    fig_spectral = go.Figure()
    fig_spectral.add_trace(
        go.Scatter(x=wavelengths_nm_sweep, y=qe_spectral, mode='lines+markers', name='Quantum Efficiency',
                   marker_color='purple'))
    fig_spectral.update_layout(title='Spectral Response / Quantum Efficiency',
                               xaxis_title='Wavelength (nm)',
                               yaxis_title='Quantum Efficiency (QE)',
                               yaxis=dict(range=[0, 1.05]))
    fig_spectral.show()


# ==============================================================================
#                      6. MAIN EXECUTION BLOCK
# ==============================================================================
def main():
    """Main function to run the simulation flow."""
    # --- Initial Setup ---
    setup_device(device, mesh_file)
    define_physical_models(device, region, material)
    define_doping_profile(device, region)
    set_initial_conditions_and_contacts(device, region)

    # --- Initial Solve ---
    devsim.set_parameter(name="cathode_bias", value=0.0)
    devsim.set_parameter(name="anode_bias", value=0.0)
    devsim.solve(type="dc", absolute_error=1.0, relative_error=1e-14, maximum_iterations=100)

    # --- Extract Doping Profile for Plotting ---
    y_coords = devsim.get_node_model_values(device=device, region=region, name="y")
    doping_data = {
        'y': y_coords,
        'N_a': devsim.get_node_model_values(device=device, region=region, name="N_a"),
        'N_d': devsim.get_node_model_values(device=device, region=region, name="N_d"),
        'NetDoping': devsim.get_node_model_values(device=device, region=region, name="NetDoping")
    }

    # --- Run Sweeps ---
    print("Running dark I-V sweep...")
    voltages, currents = run_dark_iv_sweep(device)

    print("Running optical simulation for QE...")
    wavelengths_nm_sweep, qe_spectral = run_optical_simulation(device, material)

    print("Running C-V sweep...")
    cv_voltages, capacitances = run_cv_sweep(device)

    # --- Plot Results ---
    print("Generating plots...")
    plot_results(voltages, currents, doping_data, cv_voltages, capacitances, wavelengths_nm_sweep, qe_spectral)

    print("Simulation finished.")


if __name__ == "__main__":
    main()