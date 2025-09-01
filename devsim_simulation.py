# devsim_simulation.py
# Final corrected script with realistic SRV values and fixed impact ionization

import devsim
import os
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ==============================================================================
#                      SETUP GLOBAL PARAMETERS
# ==============================================================================
device = "photodiode"
mesh_file = "output/photodiode_mesh.msh"
photon_flux = 0.0
peak_p_doping = 5e17
junction_depth = 0.5
doping_straggle = 0.1
TEMPERATURE_K = 300.0
# CORRECTED: Use more realistic SRV values for a passivated surface
s_n = 100.0  # Electron SRV at the top surface (cm/s)
s_p = 100.0  # Hole SRV at the top surface (cm/s)
WAVELENGTH_START_NM = 400
WAVELENGTH_END_NM = 1100
WAVELENGTH_POINTS = 71

def get_silicon_optical_constants_lookup(wavelength_nm):
    """
    Calculates n_r and k_e for Si using a lookup table and interpolation.
    Data is from sources like Palik, "Handbook of Optical Constants of Solids".
    """
    # Data table: Wavelength (nm), n_r, k_e
    # A small subset of data for demonstration
    wl_data = np.array([400, 450, 500, 550, 600, 650, 700, 800, 900, 1000])
    n_data = np.array([5.57, 4.67, 4.29, 4.07, 3.93, 3.83, 3.76, 3.66, 3.60, 3.56])
    k_data = np.array([0.38, 0.116, 0.056, 0.03, 0.02, 0.012, 0.008, 0.004, 0.002, 0.001])

    # Use numpy's interpolation function to find values for the given wavelength
    n_r = np.interp(wavelength_nm, wl_data, n_data)
    k_e = np.interp(wavelength_nm, wl_data, k_data)

    return (n_r, k_e)


def get_alpha_for_wavelength(wavelength_nm):
    """
    Calculates alpha from the extinction coefficient k_e.
    This is now physically consistent.
    """
    n_r, k_e = get_silicon_optical_constants_lookup(wavelength_nm)

    # alpha = 4 * pi * k_e / lambda
    alpha_cm = (4 * np.pi * k_e) / (wavelength_nm * 1e-7)  # wavelength converted to cm
    return alpha_cm


def get_reflectivity(wavelength_nm, use_arc=False, n_arc=2.0, d_arc_nm=75.0):
    """
    Calculates reflectivity using the new, accurate optical constants.
    """
    n_air = 1.0
    # This function now calls our new, accurate lookup function
    n_si, k_si = get_silicon_optical_constants_lookup(wavelength_nm)
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
# STEP 1: INITIALIZATION AND MESH LOADING
# ==============================================================================
print(f"Loading mesh: {mesh_file}")
if not os.path.exists(mesh_file):
    raise FileNotFoundError(f"Mesh file not found at '{mesh_file}'. Please run create_mesh.py first.")
devsim.create_gmsh_mesh(mesh=device, file=mesh_file)
devsim.add_gmsh_region(mesh=device, gmsh_name="p_region", region="p_region", material="Silicon")
devsim.add_gmsh_region(mesh=device, gmsh_name="n_region", region="n_region", material="Silicon")
devsim.add_gmsh_contact(mesh=device, gmsh_name="anode", region="p_region", name="anode", material="metal")
devsim.add_gmsh_contact(mesh=device, gmsh_name="cathode", region="n_region", name="cathode", material="metal")
devsim.add_gmsh_interface(mesh=device, gmsh_name="pn_junction", region0="p_region", region1="n_region",
                          name="pn_junction")
devsim.finalize_mesh(mesh=device)
devsim.create_device(mesh=device, device=device)
print("\n--- Step 1 complete: Mesh loading and device creation ---")

# --- VERIFICATION for Step 1 ---
print("\n--- Running Verification Checks for Step 1 ---")
try:
    device_list = devsim.get_device_list()
    region_list = devsim.get_region_list(device=device)
    contact_list = devsim.get_contact_list(device=device)
    interface_list = devsim.get_interface_list(device=device)
    if (len(device_list) == 1 and len(region_list) == 2 and len(contact_list) == 2 and len(interface_list) == 1):
        print("✅ Verification PASSED: Device structure (regions, contacts, interfaces) is correct.")
    else:
        print("❌ Verification FAILED: The device structure is not as expected.")
except devsim.error as msg:
    print(f"❌ An error occurred during Step 1 verification: {msg}")


# ==============================================================================
# STEP 2: DEFINING PHYSICS AND MATERIAL PROPERTIES
# ==============================================================================
def set_silicon_parameters(device, region):
    """Sets the basic material parameters for Silicon, including temperature dependence."""
    # Physical constants
    devsim.set_parameter(name="k_B_eV", value=8.617e-5)  # Boltzmann constant in eV/K
    devsim.set_parameter(name="Eg", value=1.12)  # Silicon bandgap in eV
    devsim.set_parameter(name="Nc_300K", value=2.8e19)  # Conduction band density of states at 300K

    # Global device parameters
    devsim.set_parameter(name="T", value=TEMPERATURE_K)  # Set temperature from global variable
    devsim.set_parameter(device=device, region=region, name="Permittivity", value=11.9 * 8.854e-14)
    devsim.set_parameter(device=device, region=region, name="ElectronCharge", value=1.6e-19)

    # Temperature-dependent Intrinsic Carrier Density Model
    ni_model_eq = "Nc_300K * pow(T/300.0, 1.5) * exp(-Eg / (2.0 * k_B_eV * T))"
    devsim.node_model(device=device, region=region, name="IntrinsicCarrierDensity", equation=ni_model_eq)

def define_tat_model(device, region):
    """
    Defines the Hurkx Trap-Assisted Tunneling (TAT) model.
    This is a field-enhanced generation mechanism, important for reverse-bias leakage.
    """
    # Parameters for the Hurkx TAT model in Silicon
    # N_t: Trap density (cm^-3) - concentration of mid-gap defects
    # E_t: Trap energy level relative to intrinsic level (eV)
    # gamma: Model fitting parameter (unitless)
    # delta: Model fitting parameter (unitless)
    devsim.set_parameter(device=device, region=region, name="N_t_TAT", value=1e15) # Trap density
    devsim.set_parameter(device=device, region=region, name="E_t_TAT", value=0.0)   # Mid-gap traps
    devsim.set_parameter(device=device, region=region, name="gamma_TAT", value=2.5)
    devsim.set_parameter(device=device, region=region, name="delta_TAT", value=3.0)

    # The field enhancement factor Gamma_TAT. It depends on the local electric field.
    # We re-use the absolute electric field at the nodes calculated for impact ionization.
    gamma_eq = "gamma_TAT * pow(E_mag_node_abs / 1e6, delta_TAT)"
    devsim.node_model(device=device, region=region, name="Gamma_TAT", equation=gamma_eq)

    # Hurkx TAT Generation Rate (G_TAT)
    # This is essentially the SRH rate multiplied by the field enhancement factor.
    tat_generation_eq = "( (Electrons*Holes - n_i_squared) / (taup*(Electrons + IntrinsicCarrierDensity) + taun*(Holes + IntrinsicCarrierDensity)) ) * Gamma_TAT"
    devsim.node_model(device=device, region=region, name="G_TAT", equation=tat_generation_eq)

    print(f"    Defined Trap-Assisted Tunneling (TAT) model for region: {region}")

def define_srh_lifetime_models(device, region):
    """
    Defines doping-dependent SRH lifetimes using the Klaassen model.
    Lifetime decreases in heavily doped regions.
    """
    # Parameters for the doping-dependent lifetime model
    # tau_max: lifetime in intrinsic material
    # N_ref: reference doping concentration where lifetime starts to degrade
    tau_max_n, N_ref_n = 1.0e-6, 5.0e16
    tau_max_p, N_ref_p = 1.0e-6, 5.0e16

    # Equation for electron lifetime (taun)
    eqn_n = f"{tau_max_n} / (1 + TotalDoping / {N_ref_n})"
    devsim.node_model(device=device, region=region, name="taun", equation=eqn_n)

    # Equation for hole lifetime (taup)
    eqn_p = f"{tau_max_p} / (1 + TotalDoping / {N_ref_p})"
    devsim.node_model(device=device, region=region, name="taup", equation=eqn_p)

def define_doping(device, p_peak, j_depth_cm, p_straggle_cm, n_bulk):
    """
    Defines a Gaussian profile for the p-type implant and uniform n-type substrate.
    Note: The junction is at y=0, so the peak of the implant is at y=-junction_depth.
    """
    # --- p_region (Gaussian implant) ---
    gaussian_p_eq = f"{p_peak} * exp(-0.5 * ((y + {j_depth_cm}) / {p_straggle_cm})^2)"

    devsim.node_model(device=device, region="p_region", name="Acceptors", equation=gaussian_p_eq)
    devsim.node_model(device=device, region="p_region", name="Donors", equation="0.0")

    # --- n_region (Uniform bulk) ---
    devsim.node_model(device=device, region="n_region", name="Acceptors", equation="0.0")
    devsim.node_model(device=device, region="n_region", name="Donors", equation=f"{n_bulk}")

    # --- NetDoping for both regions ---
    for region in ["p_region", "n_region"]:
        devsim.node_model(device=device, region=region, name="NetDoping", equation="Donors - Acceptors")

    print(f"Defined Gaussian p-implant (peak={p_peak:.1e}) and uniform n-bulk (N_D={n_bulk:.1e})")


def define_mobility_models(device, region):
    """
    Defines mobility using the Caughey-Thomas model, including both
    doping-dependence (low-field) and field-dependence (high-field velocity saturation).
    """
    # --- Part 1: Doping-Dependent Low-Field Mobility (at nodes) ---
    devsim.node_model(device=device, region=region, name="TotalDoping", equation="abs(Acceptors) + abs(Donors)")

    # Caughey-Thomas parameters for low-field mobility
    mu_max_n, mu_min_n, N_ref_n, alpha_n = 1417.0, 68.5, 1.10e17, 0.711
    low_field_eqn_n = f"{mu_min_n} + ({mu_max_n} - {mu_min_n}) / (1 + (TotalDoping / {N_ref_n})^{alpha_n})"
    devsim.node_model(device=device, region=region, name="LowFieldElectronMobility", equation=low_field_eqn_n)

    mu_max_p, mu_min_p, N_ref_p, alpha_p = 470.5, 44.9, 2.23e17, 0.719
    low_field_eqn_p = f"{mu_min_p} + ({mu_max_p} - {mu_min_p}) / (1 + (TotalDoping / {N_ref_p})^{alpha_p})"
    devsim.node_model(device=device, region=region, name="LowFieldHoleMobility", equation=low_field_eqn_p)

    # --- Part 2: Field-Dependent High-Field Mobility (at edges) ---
    # Average the low-field mobility from nodes to edges
    devsim.edge_average_model(device=device, region=region, node_model="LowFieldElectronMobility",
                              edge_model="LowFieldElectronMobility_edge")
    devsim.edge_average_model(device=device, region=region, node_model="LowFieldHoleMobility",
                              edge_model="LowFieldHoleMobility_edge")

    # Get the magnitude of the driving electric field along the edge
    # E_parallel = abs( (V@n0 - V@n1)/h ) = abs(ElectricField)
    devsim.edge_from_node_model(device=device, region=region, node_model="Potential")
    devsim.edge_model(device=device, region=region, name="ElectricField",
                      equation="(Potential@n0 - Potential@n1) * EdgeInverseLength")

    devsim.edge_model(device=device, region=region, name="EParallel", equation="abs(ElectricField)")

    # Saturation velocity parameters for Silicon
    v_sat_n, beta_n = 1.07e7, 2.0
    v_sat_p, beta_p = 8.37e6, 1.0

    # Final Caughey-Thomas mobility formula implemented on the edges
    final_eqn_n = f"LowFieldElectronMobility_edge / (1 + (LowFieldElectronMobility_edge * EParallel / {v_sat_n})^{beta_n})^(1/{beta_n})"
    devsim.edge_model(device=device, region=region, name="ElectronMobility", equation=final_eqn_n)

    final_eqn_p = f"LowFieldHoleMobility_edge / (1 + (LowFieldHoleMobility_edge * EParallel / {v_sat_p})^{beta_p})^(1/{beta_p})"
    devsim.edge_model(device=device, region=region, name="HoleMobility", equation=final_eqn_p)




print("\nSetting silicon material parameters...")
set_silicon_parameters(device=device, region="p_region")
set_silicon_parameters(device=device, region="n_region")
define_doping(device=device,
              p_peak=peak_p_doping,
              j_depth_cm=junction_depth * 1e-4,
              p_straggle_cm=doping_straggle * 1e-4,
              n_bulk=1e15)


for region in ["p_region", "n_region"]:
    devsim.node_solution(device=device, region=region, name="Potential")
    devsim.node_solution(device=device, region=region, name="Electrons")
    devsim.node_solution(device=device, region=region, name="Holes")

print("Defining doping-dependent mobility models...")
define_mobility_models(device=device, region="p_region")
define_mobility_models(device=device, region="n_region")

print("Defining doping-dependent SRH lifetime models...")
define_srh_lifetime_models(device=device, region="p_region")
define_srh_lifetime_models(device=device, region="n_region")

print("Defining Trap-Assisted Tunneling (TAT) models...")
define_tat_model(device=device, region="p_region")
define_tat_model(device=device, region="n_region")

print("\n--- Step 2 complete: Physics and doping defined ---")

# ==============================================================================
# STEP 3: SETTING UP THE PHOTODIODE PHYSICAL MODEL AND EQUATIONS
# ==============================================================================
print("--- STEP 3: Setting Up Full Photodiode Physical Model ---")

# --- Part A: Create Solution Variables ---


# --- Part B: Define ALL Bulk Physical Models ---
devsim.set_parameter(name="PhotonFlux", value=photon_flux)

# ADD THESE LINES HERE
# Calculate ThermalVoltage in Python and set it as a global devsim parameter
k_B_eV = devsim.get_parameter(name="k_B_eV")
T = devsim.get_parameter(name="T")
devsim.set_parameter(name="ThermalVoltage", value=k_B_eV * T)

for region in ["p_region", "n_region"]:
    # The incorrect devsim.node_model line for ThermalVoltage has been removed from the loop.

    devsim.edge_model(device=device, region=region, name="DField", equation="Permittivity * ElectricField")

    devsim.edge_model(device=device, region=region, name="DField", equation="Permittivity * ElectricField")
    devsim.edge_model(device=device, region=region, name="DField:Potential@n0",
                      equation="Permittivity * EdgeInverseLength")
    devsim.edge_model(device=device, region=region, name="DField:Potential@n1",
                      equation="-Permittivity * EdgeInverseLength")
    devsim.node_model(device=device, region=region, name="SpaceCharge",
                      equation="ElectronCharge * (Holes - Electrons + NetDoping)")
    devsim.edge_model(device=device, region=region, name="vdiff",
                      equation="(Potential@n0 - Potential@n1)/ThermalVoltage")
    devsim.edge_model(device=device, region=region, name="Bernoulli_vdiff", equation="B(vdiff)")
    devsim.edge_model(device=device, region=region, name="Bernoulli_neg_vdiff", equation="B(-vdiff)")
    devsim.edge_from_node_model(device=device, region=region, node_model="Electrons")
    devsim.edge_from_node_model(device=device, region=region, node_model="Holes")

    electron_current_eq = "ElectronCharge * ElectronMobility * ThermalVoltage * EdgeInverseLength * (Electrons@n1 * Bernoulli_neg_vdiff - Electrons@n0 * Bernoulli_vdiff)"
    devsim.edge_model(device=device, region=region, name="ElectronCurrent", equation=electron_current_eq)

    for v in ["Potential", "Electrons", "Holes"]:
        for n in ["n0", "n1"]:
            devsim.edge_model(device=device, region=region, name=f"ElectronCurrent:{v}@{n}",
                              equation=f"diff({electron_current_eq}, {v}@{n})")

    hole_current_eq = "ElectronCharge * HoleMobility * ThermalVoltage * EdgeInverseLength * (Holes@n1 * Bernoulli_vdiff - Holes@n0 * Bernoulli_neg_vdiff)"
    devsim.edge_model(device=device, region=region, name="HoleCurrent", equation=hole_current_eq)

    for v in ["Potential", "Electrons", "Holes"]:
        for n in ["n0", "n1"]:
            devsim.edge_model(device=device, region=region, name=f"HoleCurrent:{v}@{n}",
                              equation=f"diff({hole_current_eq}, {v}@{n})")

    devsim.node_model(device=device, region=region, name="n_i_squared", equation="IntrinsicCarrierDensity^2")
    devsim.node_model(device=device, region=region, name="IntrinsicElectrons",
                      equation="0.5*(NetDoping+(NetDoping^2+4*n_i_squared)^0.5)")
    devsim.node_model(device=device, region=region, name="IntrinsicHoles",
                      equation="0.5*(-NetDoping+(NetDoping^2+4*n_i_squared)^0.5)")

    # --- 1. Define Explicit SRH Trap Parameters ---
    # These are the fundamental inputs for recombination.
    # These values are typical for silicon with mid-gap traps.
    Nt = 1e10  # Trap density in cm^-3
    sigma_n = 1e-15  # Electron capture cross-section in cm^2
    sigma_p = 1e-15  # Hole capture cross-section in cm^2

    # --- 2. Calculate Thermal Velocity and Lifetimes ---
    v_th = 1e7  # Thermal velocity in cm/s at 300K
    tau_n_srh = 1.0 / (sigma_n * v_th * Nt)
    tau_p_srh = 1.0 / (sigma_p * v_th * Nt)

    # --- 3. Implement the Full SRH Model in devsim ---
    # For mid-gap traps, n1 = p1 = n_i
    devsim.node_model(device=device, region=region, name="n1_srh", equation="IntrinsicCarrierDensity")
    devsim.node_model(device=device, region=region, name="p1_srh", equation="IntrinsicCarrierDensity")

    # Set the calculated lifetimes as parameters in devsim
    devsim.set_parameter(name="tau_n_srh", value=tau_n_srh)
    devsim.set_parameter(name="tau_p_srh", value=tau_p_srh)

    # The full SRH recombination rate equation
    srh_expression = "(Electrons * Holes - n_i_squared) / (tau_p_srh * (Electrons + n1_srh) + tau_n_srh * (Holes + p1_srh))"
    devsim.node_model(name="USRH",device=device, region=region, equation=srh_expression)

    # Also redefine the OpticalGeneration model here
    devsim.node_model(device=device, region=region, name="OpticalGeneration",
                      equation="PhotonFlux * alpha * exp(-alpha * abs(y))")

    # CRITICAL: Initialize G_impact as zero here (will be updated later)
    devsim.node_model(device=device, region=region, name="G_impact", equation="0.0")
    devsim.node_model(device=device, region=region, name="G_TAT", equation="0.0")

    # Define the full NetRecombination equation in a single string for consistency
    net_recombination_eq = "USRH - OpticalGeneration - G_impact - G_TAT"

    # Use the full string to define the model
    devsim.node_model(device=device, region=region, name="NetRecombination", equation=net_recombination_eq)

    # Define the derivatives
    devsim.node_model(device=device, region=region, name="NetRecombination:Electrons",
                      equation=f"diff({net_recombination_eq}, Electrons)")
    devsim.node_model(device=device, region=region, name="NetRecombination:Holes",
                      equation=f"diff({net_recombination_eq}, Holes)")
    devsim.node_model(device=device, region=region, name="eCharge_x_NetRecomb",
                      equation="ElectronCharge * NetRecombination")
    devsim.node_model(device=device, region=region, name="eCharge_x_NetRecomb:Electrons",
                      equation="ElectronCharge * NetRecombination:Electrons")
    devsim.node_model(device=device, region=region, name="eCharge_x_NetRecomb:Holes",
                      equation="ElectronCharge * NetRecombination:Holes")

    devsim.node_model(device=device, region=region, name="Neg_eCharge_x_NetRecomb",
                      equation="-ElectronCharge * NetRecombination")
    devsim.node_model(device=device, region=region, name="Neg_eCharge_x_NetRecomb:Electrons",
                      equation="-ElectronCharge * NetRecombination:Electrons")
    devsim.node_model(device=device, region=region, name="Neg_eCharge_x_NetRecomb:Holes",
                      equation="-ElectronCharge * NetRecombination:Holes")

# Define Impact Ionization Models (Selberherr Model for Silicon)
print("    Defining impact ionization models...")
for region in ["p_region", "n_region"]:
    # Parameters for Selberherr model in Silicon
    devsim.set_parameter(device=device, region=region, name="a_n", value=7.03e5)
    devsim.set_parameter(device=device, region=region, name="b_n", value=1.231e6)
    devsim.set_parameter(device=device, region=region, name="a_p", value=1.582e6)
    devsim.set_parameter(device=device, region=region, name="b_p", value=2.036e6)

# After the main models are set up, update G_impact with actual physics
print("    Updating impact ionization with field-dependent models...")
for region in ["p_region", "n_region"]:
    # 1. Create placeholder node models FIRST
    devsim.node_model(device=device, region=region, name="E_mag_node", equation="0.0")
    devsim.node_model(device=device, region=region, name="Jn_node", equation="0.0")
    devsim.node_model(device=device, region=region, name="Jp_node", equation="0.0")

    # 2. Now populate them with edge_average_model
    devsim.edge_average_model(device=device, region=region,
                              edge_model="ElectricField",
                              node_model="E_mag_node",
                              average_type="arithmetic")

    # 3. Take absolute value at nodes
    devsim.node_model(device=device, region=region, name="E_mag_node_abs",
                      equation="abs(E_mag_node)")

    # 4. Define Ionization Coefficients - CORRECTED SYNTAX
    # Use division operator instead of negative exponent
    devsim.node_model(device=device, region=region, name="alpha_n_node",
                      equation="ifelse(E_mag_node_abs > 1.0, a_n * exp(-b_n / (E_mag_node_abs + 1e-20)), 0.0)")
    devsim.node_model(device=device, region=region, name="alpha_p_node",
                      equation="ifelse(E_mag_node_abs > 1.0, a_p * exp(-b_p / (E_mag_node_abs + 1e-20)), 0.0)")

    # 5. Populate current node models
    devsim.edge_average_model(device=device, region=region,
                              edge_model="ElectronCurrent",
                              node_model="Jn_node",
                              average_type="arithmetic")
    devsim.edge_average_model(device=device, region=region,
                              edge_model="HoleCurrent",
                              node_model="Jp_node",
                              average_type="arithmetic")

    # 6. Get magnitudes
    devsim.node_model(device=device, region=region, name="Jn_mag_node",
                      equation="abs(Jn_node)")
    devsim.node_model(device=device, region=region, name="Jp_mag_node",
                      equation="abs(Jp_node)")

    # 7. Update G_impact with the actual generation rate
    generation_eq = "(alpha_n_node * Jn_mag_node + alpha_p_node * Jp_mag_node) / ElectronCharge"
    devsim.node_model(device=device, region=region, name="G_impact", equation=generation_eq)
# --- Part C: Define ALL Boundary Condition Models ---
print("  3C: Defining all boundary condition models...")

# Define SRV parameters and models specifically for the anode
devsim.set_parameter(device=device, name="Sn", value=s_n)
devsim.set_parameter(device=device, name="Sp", value=s_p)

# Electron recombination current at the anode surface
devsim.contact_node_model(
    device=device, contact="anode", name="ElectronSurfaceRecombinationCurrent",
    equation="ElectronCharge * Sn * (Electrons - IntrinsicElectrons)"
)
devsim.contact_node_model(
    device=device, contact="anode", name="ElectronSurfaceRecombinationCurrent:Electrons",
    equation="ElectronCharge * Sn"
)

# Hole recombination current at the anode surface
devsim.contact_node_model(
    device=device, contact="anode", name="HoleSurfaceRecombinationCurrent",
    equation="ElectronCharge * Sp * (Holes - IntrinsicHoles)"
)
devsim.contact_node_model(
    device=device, contact="anode", name="HoleSurfaceRecombinationCurrent:Holes",
    equation="ElectronCharge * Sp"
)

for contact in ["anode", "cathode"]:
    devsim.set_parameter(device=device, name=f"{contact}_bias", value=0.0)
    devsim.contact_node_model(device=device, contact=contact, name=f"{contact}_potential_bc",
                              equation=f"Potential - {contact}_bias")
    devsim.contact_node_model(device=device, contact=contact, name=f"{contact}_potential_bc:Potential", equation="1.0")
    devsim.contact_node_model(device=device, contact=contact, name=f"{contact}_electrons_bc",
                              equation="Electrons - IntrinsicElectrons")
    devsim.contact_node_model(device=device, contact=contact, name=f"{contact}_electrons_bc:Electrons", equation="1.0")
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

# --- Part D: Solve for Initial Equilibrium (Staged Method with Correct Initial Guess) ---
print("  3D: Solving for initial equilibrium state (two-step method)...")
print("    Creating robust initial guess based on charge neutrality...")

for region in ["p_region", "n_region"]:
    devsim.set_node_values(device=device, region=region, name="Electrons", init_from="IntrinsicElectrons")
    devsim.set_node_values(device=device, region=region, name="Holes", init_from="IntrinsicHoles")
    devsim.node_model(device=device, region=region, name="InitialPotential",
                      equation="ThermalVoltage * log(IntrinsicElectrons/IntrinsicCarrierDensity)")
    devsim.set_node_values(device=device, region=region, name="Potential", init_from="InitialPotential")

print("    Step 1/2: Assembling and solving for Potential only...")
# FIRST: Setup Potential equation in regions
for region in ["p_region", "n_region"]:
    devsim.equation(device=device, region=region, name="PotentialEquation",
                    variable_name="Potential",
                    node_model="SpaceCharge",
                    edge_model="DField",
                    variable_update="log_damp")

# Setup contact equations for Potential (without edge_charge_model initially)
for contact in ["anode", "cathode"]:
    devsim.contact_equation(device=device, contact=contact, name="PotentialEquation",
                            node_model=f"{contact}_potential_bc")

# Setup interface equation for Potential
devsim.interface_equation(device=device, interface="pn_junction", name="PotentialEquation",
                          interface_model="Potential_continuity", type="continuous")

# Solve for Potential only
devsim.solve(type="dc", absolute_error=1e-10, relative_error=1e-12, maximum_iterations=100)

print("    Updating carrier guess using Boltzmann statistics...")
for region in ["p_region", "n_region"]:
    devsim.node_model(device=device, region=region, name="UpdatedElectrons",
                      equation="IntrinsicCarrierDensity*exp(Potential/ThermalVoltage)")
    devsim.node_model(device=device, region=region, name="UpdatedHoles",
                      equation="IntrinsicCarrierDensity*exp(-Potential/ThermalVoltage)")
    devsim.set_node_values(device=device, region=region, name="Electrons", init_from="UpdatedElectrons")
    devsim.set_node_values(device=device, region=region, name="Holes", init_from="UpdatedHoles")

print("    Step 2/2: Assembling continuity equations and solving the full system...")
# SECOND: Setup continuity equations in regions
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

# Setup interface equations for continuity
devsim.interface_equation(device=device, interface="pn_junction", name="ElectronContinuityEquation",
                          interface_model="Electrons_continuity", type="continuous")
devsim.interface_equation(device=device, interface="pn_junction", name="HoleContinuityEquation",
                          interface_model="Holes_continuity", type="continuous")

print("    Setting up CATHODE boundary conditions (Ohmic)...")
devsim.contact_equation(device=device, contact="cathode", name="PotentialEquation",
                        node_model="cathode_potential_bc", edge_charge_model="DField")
devsim.contact_equation(device=device, contact="cathode", name="ElectronContinuityEquation",
                        node_model="cathode_electrons_bc")
devsim.contact_equation(device=device, contact="cathode", name="HoleContinuityEquation",
                        node_model="cathode_holes_bc")

# --- ANODE: Contact with Surface Recombination (Robin BCs for carriers) ---
print("    Setting up ANODE boundary conditions (with SRV)...")
devsim.contact_equation(device=device, contact="anode", name="PotentialEquation",
                        node_model="anode_potential_bc", edge_charge_model="DField")
devsim.contact_equation(device=device, contact="anode", name="ElectronContinuityEquation",
                        node_model="ElectronSurfaceRecombinationCurrent",
                        edge_current_model="ElectronCurrent")
devsim.contact_equation(device=device, contact="anode", name="HoleContinuityEquation",
                        node_model="HoleSurfaceRecombinationCurrent",
                        edge_current_model="HoleCurrent")

print("\n✅ Step 3 complete: Full photodiode model is defined and solved for equilibrium.")


# ==============================================================================
#                      HELPER FUNCTIONS FOR CHARACTERIZATION
# ==============================================================================

def run_iv_sweep(device, voltages, p_flux):
    currents = []
    devsim.set_parameter(name="PhotonFlux", value=p_flux)
    if p_flux > 0:
        print(f"DEBUG: PhotonFlux set to {devsim.get_parameter(name='PhotonFlux')}")

    for v in voltages:
        print(f"\nSetting Anode Bias: {v:.3f} V")
        devsim.set_parameter(device=device, name="anode_bias", value=v)
        try:
            devsim.solve(type="dc", absolute_error=1e10, relative_error=0.001,
                         maximum_iterations=400,
                         maximum_divergence=10)
            e_current = devsim.get_contact_current(device=device, contact="anode",
                                                   equation="ElectronContinuityEquation")
            h_current = devsim.get_contact_current(device=device, contact="anode", equation="HoleContinuityEquation")
            currents.append(e_current + h_current)
            print(f"✅ V = {v:.3f} V, Current = {currents[-1]:.4e} A/cm")
        except devsim.error as msg:
            print(f"❌ CONVERGENCE FAILED at V = {v:.3f} V. Error: {msg}")
            currents.append(float('nan'))
            break

    devsim.set_parameter(device=device, name="anode_bias", value=0.0)
    return np.array(currents)


def calculate_qe(dark_currents, light_currents, p_flux, device_width_cm, wavelength_nm):
    """
    Calculates the External Quantum Efficiency (QE) for the photodiode.
    """
    q = 1.602e-19  # Elementary charge in Coulombs
    photocurrent_density = np.abs(light_currents - dark_currents)
    electrons_per_sec_per_cm2 = photocurrent_density / q
    photons_per_sec_per_cm2 = p_flux
    qe = (electrons_per_sec_per_cm2 / photons_per_sec_per_cm2) * 100.0
    return qe


def run_cv_sweep(device, voltages, freq_hz):
    """
    Calculates C-V using numerical differentiation of charge.
    """
    capacitances = []
    DELTA_V = 0.001  # 1 mV step for numerical differentiation

    print(f"\nStarting C-V sweep...")

    devsim.set_parameter(device=device, name="anode_bias", value=voltages[0])
    devsim.solve(type="dc", absolute_error=10.0, relative_error=1e-3, maximum_iterations=300)

    for i, v in enumerate(voltages):
        print(f"Step {i + 1}/{len(voltages)}: Bias = {v:.2f} V")
        try:
            devsim.set_parameter(device=device, name="anode_bias", value=v - DELTA_V / 2.0)
            devsim.solve(type="dc", absolute_error=10.0, relative_error=1e-3, maximum_iterations=200)
            q1 = devsim.get_contact_charge(device=device, contact="anode", equation="PotentialEquation")

            devsim.set_parameter(device=device, name="anode_bias", value=v + DELTA_V / 2.0)
            devsim.solve(type="dc", absolute_error=10.0, relative_error=1e-3, maximum_iterations=100)
            q2 = devsim.get_contact_charge(device=device, contact="anode", equation="PotentialEquation")

            C = abs(q2 - q1) / DELTA_V
            capacitances.append(C)
            print(f"  C = {C * 1e12:.3f} pF/cm (q1={q1:.3e} C/cm, q2={q2:.3e} C/cm)")
        except devsim.error as msg:
            print(f"  Failed at V = {v:.2f} V: {msg}")
            capacitances.append(float('nan'))

    devsim.set_parameter(device=device, name="anode_bias", value=0.0)
    return np.array(capacitances)


def run_cv_sweep_ac(device, voltages, freq_hz):
    """
    Calculates C-V using the more robust small-signal AC analysis method.
    """
    capacitances = []
    omega = 2.0 * np.pi * freq_hz

    print(f"\nStarting AC C-V sweep at {freq_hz / 1e6:.1f} MHz...")

    for i, v in enumerate(voltages):
        devsim.set_parameter(device=device, name="anode_bias", value=v)
        print(f"Step {i + 1}/{len(voltages)}: Bias = {v:.2f} V")
        try:
            devsim.solve(type="dc", absolute_error=100.0, relative_error=3e-2, maximum_iterations=200)
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


# ==============================================================================
#                      MAIN SIMULATION AND VISUALIZATION
# ==============================================================================
if __name__ == "__main__":
    print("\n--- Running Single-Point Simulations for I-V, C-V, and QE vs. V plots ---")

    # --- I-V and QE vs. V Simulation ---
    iv_voltages = np.linspace(0, -30, 61)
    LIGHT_PHOTON_FLUX = 1e17
    WAVELENGTH_NM_SINGLE = 650

    single_alpha = get_alpha_for_wavelength(WAVELENGTH_NM_SINGLE)
    devsim.set_parameter(name="alpha", value=single_alpha)
    print(f"Using alpha = {single_alpha:.2e} 1/cm for single-point simulation at {WAVELENGTH_NM_SINGLE} nm")

    dark_currents_single = run_iv_sweep(device, iv_voltages, p_flux=0.0)
    light_currents_single = run_iv_sweep(device, iv_voltages, p_flux=LIGHT_PHOTON_FLUX)
    qe_vs_voltage = calculate_qe(dark_currents_single, light_currents_single, LIGHT_PHOTON_FLUX, device_width_cm=1.0,
                                 wavelength_nm=WAVELENGTH_NM_SINGLE)

    # # --- C-V Simulation ---
    # cv_voltages = np.linspace(0, -5, 21)
    # capacitances = run_cv_sweep_ac(device, cv_voltages, freq_hz=1e6)
    #
    # # ==============================================================================
    # #                      RUN SPECTRAL SWEEP FOR NEW PLOT
    # # ==============================================================================
    # print("\n--- STARTING SPECTRAL SWEEP for QE vs. Wavelength plot ---")
    #
    # # Check if the dark current sweep completed successfully
    # if len(dark_currents_single) != len(iv_voltages):
    #     print("WARNING: I-V sweep did not complete. Truncating voltage range for interpolation.")
    #     # Find the last valid index
    #     valid_indices = ~np.isnan(dark_currents_single)
    #     if np.any(valid_indices):
    #         last_valid = np.where(valid_indices)[0][-1]
    #         iv_voltages = iv_voltages[:last_valid + 1]
    #         dark_currents_single = dark_currents_single[:last_valid + 1]
    #     else:
    #         print("ERROR: No valid I-V data available. Cannot proceed with spectral sweep.")
    #         import sys
    #
    #         sys.exit(1)
    #
    # # 1. Correctly define the wavelengths to sweep over
    # wavelengths_nm_sweep = np.linspace(WAVELENGTH_START_NM, WAVELENGTH_END_NM, WAVELENGTH_POINTS)
    # qe_spectral_results = []
    # QE_BIAS = -2.0
    # INCIDENT_PHOTON_FLUX = 1e17
    # USE_AR_COATING = True
    #
    # # Interpolate to find the dark current at the specific bias for QE calculation
    # dark_current_at_bias = np.interp(QE_BIAS, iv_voltages, dark_currents_single)
    #
    # # 2. Add the missing loop to perform the spectral sweep
    # print(f"\nSimulating QE at a fixed bias of {QE_BIAS} V across the spectrum...")
    # devsim.set_parameter(device=device, name="anode_bias", value=QE_BIAS)
    #
    # for wl in wavelengths_nm_sweep:
    #     # Calculate optical properties for the current wavelength
    #     alpha_val = get_alpha_for_wavelength(wl)
    #
    #     # Set these parameters in the devsim model
    #     devsim.set_parameter(name="alpha", value=alpha_val)
    #     devsim.set_parameter(name="PhotonFlux", value=INCIDENT_PHOTON_FLUX)
    #
    #     try:
    #         # Solve for the light condition at this wavelength
    #         devsim.solve(type="dc", absolute_error=1e10, relative_error=1e-2, maximum_iterations=100)
    #
    #         # Get the light current
    #         e_current = devsim.get_contact_current(device=device, contact="anode",
    #                                                equation="ElectronContinuityEquation")
    #         h_current = devsim.get_contact_current(device=device, contact="anode", equation="HoleContinuityEquation")
    #         light_current_at_bias = e_current + h_current
    #
    #         # Calculate and store the QE value
    #         qe_val = calculate_qe(dark_current_at_bias, light_current_at_bias, INCIDENT_PHOTON_FLUX, 1.0, wl)
    #         qe_spectral_results.append(qe_val)
    #         print(f"  ✅ Wavelength: {wl:.1f} nm, Photocurrent: {light_current_at_bias:.3e} A/cm, QE: {qe_val:.2f}%")
    #
    #     except devsim.error as msg:
    #         print(f"  ❌ CONVERGENCE FAILED at {wl:.1f} nm. Error: {msg}")
    #         qe_spectral_results.append(float('nan'))
    #
    # # Reset photon flux to zero after the sweep
    # devsim.set_parameter(name="PhotonFlux", value=0.0)
    #
    # print("\n--- ALL SIMULATIONS COMPLETE ---")
    #
    # # ==============================================================================
    # #                      STEP 7: VISUALIZE ALL RESULTS
    # # ==============================================================================
    # print("\n--- STEP 7: Generating All Plots ---")
    #
    # # --- PLOT 1: Original Interactive I-V Plot ---
    # fig_iv = go.Figure()
    # fig_iv.add_trace(go.Scatter(x=iv_voltages, y=np.abs(dark_currents_single), mode='lines+markers',
    #                             name='Dark Current', marker_color='red'))
    # fig_iv.add_trace(go.Scatter(x=iv_voltages, y=np.abs(light_currents_single), mode='lines+markers',
    #                             name=f'Photocurrent @ {WAVELENGTH_NM_SINGLE} nm', marker_color='blue'))
    # fig_iv.update_layout(title_text="I-V Characteristics (Interactive)",
    #                      xaxis_title="Anode Voltage (V)",
    #                      yaxis_title="Current Magnitude (A/cm)",
    #                      yaxis_type="log")
    # fig_iv.show()
    #
    # # --- PLOT 2: Original Interactive QE vs. Voltage Plot ---
    # fig_qe_v = go.Figure()
    # fig_qe_v.add_trace(go.Scatter(x=iv_voltages, y=qe_vs_voltage, mode='lines+markers',
    #                               name='QE', marker_color='green'))
    # fig_qe_v.update_layout(title_text=f"QE vs. Voltage @ {WAVELENGTH_NM_SINGLE} nm (Interactive)",
    #                        xaxis_title="Anode Voltage (V)",
    #                        yaxis_title="External Quantum Efficiency (%)",
    #                        yaxis_range=[0, 105])
    # fig_qe_v.show()
    #
    # # --- PLOT 3: Original Interactive C-V and Mott-Schottky Plot ---
    # fig_cv = make_subplots(rows=1, cols=2, subplot_titles=("C-V @ 1 MHz", "Mott-Schottky Plot"))
    # valid_cv_indices = ~np.isnan(capacitances)
    # inv_C_squared = 1.0 / (capacitances[valid_cv_indices] ** 2)
    # fig_cv.add_trace(go.Scatter(x=cv_voltages[valid_cv_indices], y=capacitances[valid_cv_indices] * 1e12,
    #                             mode='lines+markers', name='Capacitance', marker_color='magenta'), row=1, col=1)
    # fig_cv.add_trace(go.Scatter(x=cv_voltages[valid_cv_indices], y=inv_C_squared,
    #                             mode='lines+markers', name='1/C²', marker_color='darkturquoise'), row=1, col=2)
    # fig_cv.update_xaxes(title_text="Anode Voltage (V)", row=1, col=1)
    # fig_cv.update_yaxes(title_text="Capacitance (pF/cm)", row=1, col=1)
    # fig_cv.update_xaxes(title_text="Anode Voltage (V)", row=1, col=2)
    # fig_cv.update_yaxes(title_text="1/C² (F⁻²cm²)", row=1, col=2)
    # fig_cv.update_layout(title_text="Capacitance Analysis (Interactive)", showlegend=False)
    # fig_cv.show()
    #
    # # --- PLOT 4: NEW Interactive Spectral Response Plot ---
    # fig_spectral = go.Figure()
    # fig_spectral.add_trace(go.Scatter(x=wavelengths_nm_sweep, y=qe_spectral_results, mode='lines+markers',
    #                                   name=f'QE at {QE_BIAS}V', marker_color='purple'))
    # fig_spectral.update_layout(title_text=f"Photodiode Spectral Response at V_anode = {QE_BIAS}V (Interactive)",
    #                            xaxis_title="Wavelength (nm)",
    #                            yaxis_title="External Quantum Efficiency (%)",
    #                            yaxis_range=[0, 105])
    # fig_spectral.show()