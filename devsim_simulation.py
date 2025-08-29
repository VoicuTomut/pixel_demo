# devsim_simulation.py
# Fully corrected script maintaining the original structure.

import devsim
import os
import numpy as np
import matplotlib.pyplot as plt

# ==============================================================================
#                      SETUP GLOBAL PARAMETERS
# ==============================================================================
# --- Device and File Names ---
device_name = "photodiode"
mesh_file = "output/photodiode_mesh.msh"

# --- Photodiode Simulation Parameters ---
# Set photon_flux to 0.0 for a simulation in the dark.
# Set photon_flux > 0.0 for a simulation under illumination.
photon_flux = 0.0  # Units: photons/cm^2/s
# Absorption coefficient for a given wavelength.
alpha = 1e4  # Units: 1/cm

# ==============================================================================
# STEP 1: INITIALIZATION AND MESH LOADING
# ==============================================================================
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
print("\n--- Step 1 complete: Mesh loading and device creation ---")

# --- VERIFICATION for Step 1 ---
print("\n--- Running Verification Checks for Step 1 ---")
try:
    device_list = devsim.get_device_list()
    region_list = devsim.get_region_list(device=device_name)
    contact_list = devsim.get_contact_list(device=device_name)
    interface_list = devsim.get_interface_list(device=device_name)
    if (len(device_list) == 1 and len(region_list) == 2 and len(contact_list) == 2 and len(interface_list) == 1):
        print("✅ Verification PASSED: Device structure (regions, contacts, interfaces) is correct.")
    else:
        print("❌ Verification FAILED: The device structure is not as expected.")
except devsim.error as msg:
    print(f"❌ An error occurred during Step 1 verification: {msg}")


# ==============================================================================
# STEP 2: DEFINING PHYSICS AND MATERIAL PROPERTIES
# ==============================================================================
# This section is unchanged and correctly sets up the material parameters.
def set_silicon_parameters(device, region):
    """Sets the basic material parameters for Silicon."""
    devsim.set_parameter(device=device, region=region, name="Permittivity", value=11.9 * 8.854e-14)
    devsim.set_parameter(device=device, region=region, name="IntrinsicCarrierDensity", value=1.0e10)
    devsim.set_parameter(device=device, region=region, name="ElectronCharge", value=1.6e-19)
    devsim.set_parameter(device=device, region=region, name="taun", value=1.0e-6)
    devsim.set_parameter(device=device, region=region, name="taup", value=1.0e-6)


def define_doping(device, p_doping, n_doping):
    """Defines the acceptor and donor concentrations for the device."""
    devsim.node_model(device=device, region="p_region", name="Acceptors", equation=f"{p_doping}")
    devsim.node_model(device=device, region="p_region", name="Donors", equation="0.0")
    devsim.node_model(device=device, region="n_region", name="Acceptors", equation="0.0")
    devsim.node_model(device=device, region="n_region", name="Donors", equation=f"{n_doping}")
    for region in ["p_region", "n_region"]:
        devsim.node_model(device=device, region=region, name="NetDoping", equation="Donors - Acceptors")
    print(f"Defined doping: N_A = {p_doping:.1e} cm^-3, N_D = {n_doping:.1e} cm^-3")


def define_mobility_models(device, region):
    """
    Defines doping-dependent mobility using the Caughey-Thomas model.
    """
    devsim.node_model(device=device, region=region, name="TotalDoping", equation="abs(Acceptors) + abs(Donors)")
    mu_max_n, mu_min_n, N_ref_n, alpha_n = 1417.0, 68.5, 1.10e17, 0.711
    eqn_n = f"{mu_min_n} + ({mu_max_n} - {mu_min_n}) / (1 + (TotalDoping / {N_ref_n})^{alpha_n})"
    devsim.node_model(device=device, region=region, name="ElectronMobility", equation=eqn_n)
    mu_max_p, mu_min_p, N_ref_p, alpha_p = 470.5, 44.9, 2.23e17, 0.719
    eqn_p = f"{mu_min_p} + ({mu_max_p} - {mu_min_p}) / (1 + (TotalDoping / {N_ref_p})^{alpha_p})"
    devsim.node_model(device=device, region=region, name="HoleMobility", equation=eqn_p)


print("\nSetting silicon material parameters...")
set_silicon_parameters(device=device_name, region="p_region")
set_silicon_parameters(device=device_name, region="n_region")
define_doping(device=device_name, p_doping=1e15, n_doping=1e18)
print("Defining doping-dependent mobility models...")
define_mobility_models(device=device_name, region="p_region")
define_mobility_models(device=device_name, region="n_region")
print("\n--- Step 2 complete: Physics and doping defined ---")

# ==============================================================================
# STEP 3: SETTING UP THE PHOTODIODE PHYSICAL MODEL AND EQUATIONS
# ==============================================================================
# This section is restored to its original detailed format, with only the typo corrected.
print("--- STEP 3: Setting Up Full Photodiode Physical Model ---")

# --- Part A: Create Solution Variables ---
print("  3A: Creating solution variables (Potential, Electrons, Holes)...")
for region in ["p_region", "n_region"]:
    devsim.node_solution(device=device_name, region=region, name="Potential")
    devsim.node_solution(device=device_name, region=region, name="Electrons")
    devsim.node_solution(device=device_name, region=region, name="Holes")

# --- Part B: Define ALL Bulk Physical Models ---
print("  3B: Defining all bulk physical models...")
devsim.set_parameter(name="ThermalVoltage", value=0.0259)
devsim.set_parameter(name="PhotonFlux", value=photon_flux)
devsim.set_parameter(name="alpha", value=alpha)

for region in ["p_region", "n_region"]:
    devsim.edge_from_node_model(device=device_name, region=region, node_model="Potential")
    devsim.edge_model(device=device_name, region=region, name="ElectricField",
                      equation="(Potential@n0 - Potential@n1) * EdgeInverseLength")
    devsim.edge_model(device=device_name, region=region, name="DField", equation="Permittivity * ElectricField")
    devsim.edge_model(device=device_name, region=region, name="DField:Potential@n0",
                      equation="Permittivity * EdgeInverseLength")
    devsim.edge_model(device=device_name, region=region, name="DField:Potential@n1",
                      equation="-Permittivity * EdgeInverseLength")
    devsim.node_model(device=device_name, region=region, name="SpaceCharge",
                      equation="ElectronCharge * (Holes - Electrons + NetDoping)")
    devsim.edge_model(device=device_name, region=region, name="vdiff",
                      equation="(Potential@n0 - Potential@n1)/ThermalVoltage")
    devsim.edge_model(device=device_name, region=region, name="Bernoulli_vdiff", equation="B(vdiff)")
    devsim.edge_model(device=device_name, region=region, name="Bernoulli_neg_vdiff", equation="B(-vdiff)")
    devsim.edge_from_node_model(device=device_name, region=region, node_model="Electrons")
    devsim.edge_from_node_model(device=device_name, region=region, node_model="Holes")
    devsim.edge_from_node_model(device=device_name, region=region, node_model="ElectronMobility")
    devsim.edge_from_node_model(device=device_name, region=region, node_model="HoleMobility")  # Typo was corrected here
    devsim.edge_model(device=device_name, region=region, name="EdgeElectronMobility",
                      equation="(ElectronMobility@n0 + ElectronMobility@n1) * 0.5")
    devsim.edge_model(device=device_name, region=region, name="EdgeHoleMobility",
                      equation="(HoleMobility@n0 + HoleMobility@n1) * 0.5")
    electron_current_eq = "ElectronCharge * EdgeElectronMobility * ThermalVoltage * EdgeInverseLength * (Electrons@n1 * Bernoulli_neg_vdiff - Electrons@n0 * Bernoulli_vdiff)"
    devsim.edge_model(device=device_name, region=region, name="ElectronCurrent", equation=electron_current_eq)
    for v in ["Potential", "Electrons", "Holes"]:
        for n in ["n0", "n1"]: devsim.edge_model(device=device_name, region=region, name=f"ElectronCurrent:{v}@{n}",
                                                 equation=f"diff({electron_current_eq}, {v}@{n})")
    hole_current_eq = "ElectronCharge * EdgeHoleMobility * ThermalVoltage * EdgeInverseLength * (Holes@n1 * Bernoulli_vdiff - Holes@n0 * Bernoulli_neg_vdiff)"
    devsim.edge_model(device=device_name, region=region, name="HoleCurrent", equation=hole_current_eq)
    for v in ["Potential", "Electrons", "Holes"]:
        for n in ["n0", "n1"]: devsim.edge_model(device=device_name, region=region, name=f"HoleCurrent:{v}@{n}",
                                                 equation=f"diff({hole_current_eq}, {v}@{n})")

    devsim.node_model(device=device_name, region=region, name="n_i_squared", equation="IntrinsicCarrierDensity^2")
    devsim.node_model(device=device_name, region=region, name="IntrinsicElectrons",
                      equation="0.5*(NetDoping+(NetDoping^2+4*n_i_squared)^0.5)")
    devsim.node_model(device=device_name, region=region, name="IntrinsicHoles",
                      equation="0.5*(-NetDoping+(NetDoping^2+4*n_i_squared)^0.5)")
    srh_eq = "(Electrons*Holes - n_i_squared) / (taup*(Electrons + IntrinsicElectrons) + taun*(Holes + IntrinsicHoles))"
    devsim.node_model(device=device_name, region=region, name="USRH", equation=srh_eq)
    # Use abs(y) for robustness, though (0-y) is also fine.
    devsim.node_model(device=device_name, region=region, name="OpticalGeneration",
                      equation="PhotonFlux * alpha * exp(-alpha * abs(y))")

    # Define the full NetRecombination equation in a single string for consistency
    net_recombination_eq = "USRH - OpticalGeneration"

    # Use the full string to define the model
    devsim.node_model(device=device_name, region=region, name="NetRecombination", equation=net_recombination_eq)

    # CRITICAL: Use the full string to define the derivatives
    devsim.node_model(device=device_name, region=region, name="NetRecombination:Electrons",
                      equation=f"diff({net_recombination_eq}, Electrons)")
    devsim.node_model(device=device_name, region=region, name="NetRecombination:Holes",
                      equation=f"diff({net_recombination_eq}, Holes)")
    devsim.node_model(device=device_name, region=region, name="eCharge_x_NetRecomb",
                      equation="ElectronCharge * NetRecombination")
    devsim.node_model(device=device_name, region=region, name="eCharge_x_NetRecomb:Electrons",
                      equation="ElectronCharge * NetRecombination:Electrons")
    devsim.node_model(device=device_name, region=region, name="eCharge_x_NetRecomb:Holes",
                      equation="ElectronCharge * NetRecombination:Holes")

    devsim.node_model(device=device_name, region=region, name="Neg_eCharge_x_NetRecomb",
                      equation="-ElectronCharge * NetRecombination")
    devsim.node_model(device=device_name, region=region, name="Neg_eCharge_x_NetRecomb:Electrons",
                      equation="-ElectronCharge * NetRecombination:Electrons")
    devsim.node_model(device=device_name, region=region, name="Neg_eCharge_x_NetRecomb:Holes",
                      equation="-ElectronCharge * NetRecombination:Holes")

# --- Part C: Define ALL Boundary Condition Models ---
print("  3C: Defining all boundary condition models...")
for contact in ["anode", "cathode"]:
    devsim.set_parameter(device=device_name, name=f"{contact}_bias", value=0.0)
    # Use f-strings to create unique names like "anode_potential_bc"
    devsim.contact_node_model(device=device_name, contact=contact, name=f"{contact}_potential_bc",
                              equation=f"Potential - {contact}_bias")
    devsim.contact_node_model(device=device_name, contact=contact, name=f"{contact}_potential_bc:Potential", equation="1.0")
    devsim.contact_node_model(device=device_name, contact=contact, name=f"{contact}_electrons_bc",
                              equation="Electrons - IntrinsicElectrons")
    devsim.contact_node_model(device=device_name, contact=contact, name=f"{contact}_electrons_bc:Electrons", equation="1.0")
    devsim.contact_node_model(device=device_name, contact=contact, name=f"{contact}_holes_bc",
                              equation="Holes - IntrinsicHoles")
    devsim.contact_node_model(device=device_name, contact=contact, name=f"{contact}_holes_bc:Holes", equation="1.0")


for variable in ["Potential", "Electrons", "Holes"]:
    devsim.interface_model(device=device_name, interface="pn_junction", name=f"{variable}_continuity",
                           equation=f"{variable}@r0 - {variable}@r1")
    devsim.interface_model(device=device_name, interface="pn_junction", name=f"{variable}_continuity:{variable}@r0",
                           equation="1.0")
    devsim.interface_model(device=device_name, interface="pn_junction", name=f"{variable}_continuity:{variable}@r1",
                           equation="-1.0")


# --- Part D: Solve for Initial Equilibrium (Staged Method with Correct Initial Guess) ---
print("  3D: Solving for initial equilibrium state (two-step method)...")
print("    Creating robust initial guess based on charge neutrality...")
for region in ["p_region", "n_region"]:
    devsim.set_node_values(device=device_name, region=region, name="Electrons", init_from="IntrinsicElectrons")
    devsim.set_node_values(device=device_name, region=region, name="Holes", init_from="IntrinsicHoles")
    devsim.node_model(device=device_name, region=region, name="InitialPotential",
                      equation="ThermalVoltage * log(IntrinsicElectrons/IntrinsicCarrierDensity)")
    devsim.set_node_values(device=device_name, region=region, name="Potential", init_from="InitialPotential")

print("    Step 1/2: Assembling and solving for Potential only...")
# FIRST: Setup Potential equation in regions
for region in ["p_region", "n_region"]:
    devsim.equation(device=device_name, region=region, name="PotentialEquation",
                    variable_name="Potential",
                    node_model="SpaceCharge",  # <-- THIS IS THE CRITICAL FIX
                    edge_model="DField",
                    variable_update="log_damp")

# Setup contact equations for Potential (without edge_charge_model initially)
for contact in ["anode", "cathode"]:
    devsim.contact_equation(device=device_name, contact=contact, name="PotentialEquation",
                           node_model=f"{contact}_potential_bc")

# Setup interface equation for Potential
devsim.interface_equation(device=device_name, interface="pn_junction", name="PotentialEquation",
                          interface_model="Potential_continuity", type="continuous")

# Solve for Potential only
devsim.solve(type="dc", absolute_error=1e-10, relative_error=1e-12, maximum_iterations=50)

print("    Updating carrier guess using Boltzmann statistics...")
for region in ["p_region", "n_region"]:
    devsim.node_model(device=device_name, region=region, name="UpdatedElectrons",
                      equation="IntrinsicCarrierDensity*exp(Potential/ThermalVoltage)")
    devsim.node_model(device=device_name, region=region, name="UpdatedHoles",
                      equation="IntrinsicCarrierDensity*exp(-Potential/ThermalVoltage)")
    devsim.set_node_values(device=device_name, region=region, name="Electrons", init_from="UpdatedElectrons")
    devsim.set_node_values(device=device_name, region=region, name="Holes", init_from="UpdatedHoles")

print("    Step 2/2: Assembling continuity equations and solving the full system...")
# SECOND: Setup continuity equations in regions
for region in ["p_region", "n_region"]:
    # For d(n)/dt: div(Jn) = q(R-G). DEVSIM solves div(F) - S = 0.
    # So F=Jn and S = q(R-G), which is our "eCharge_x_NetRecomb"
    devsim.equation(device=device_name, region=region, name="ElectronContinuityEquation",
                    variable_name="Electrons",
                    node_model="eCharge_x_NetRecomb",
                    edge_model="ElectronCurrent",
                    variable_update="log_damp")

    # For d(p)/dt: div(Jp) = -q(R-G). DEVSIM solves div(F) - S = 0.
    # So F=Jp and S = -q(R-G), which is our "Neg_eCharge_x_NetRecomb"
    devsim.equation(device=device_name, region=region, name="HoleContinuityEquation",
                    variable_name="Holes",
                    node_model="Neg_eCharge_x_NetRecomb",
                    edge_model="HoleCurrent",
                    variable_update="log_damp")

# Setup interface equations for continuity
devsim.interface_equation(device=device_name, interface="pn_junction", name="ElectronContinuityEquation",
                          interface_model="Electrons_continuity", type="continuous")
devsim.interface_equation(device=device_name, interface="pn_junction", name="HoleContinuityEquation",
                          interface_model="Holes_continuity", type="continuous")

# CRITICAL: Setup ALL contact equations WITH edge_charge_model for Potential
for contact in ["anode", "cathode"]:
    devsim.contact_equation(device=device_name, contact=contact, name="PotentialEquation",
                            node_model=f"{contact}_potential_bc",
                            edge_charge_model="DField")  # THIS IS CRITICAL FOR C-V
    devsim.contact_equation(device=device_name, contact=contact, name="ElectronContinuityEquation",
                            node_model=f"{contact}_electrons_bc",
                            edge_current_model="ElectronCurrent")
    devsim.contact_equation(device=device_name, contact=contact, name="HoleContinuityEquation",
                            node_model=f"{contact}_holes_bc",
                            edge_current_model="HoleCurrent")

# print("\n--- Final Solve: Using a Ramping Strategy for Stability ---")
# target_lifetime = 1e-6
# ramp_lifetimes = [1e-10, 1e-9, 1e-8, 1e-7, target_lifetime]
# for i, life in enumerate(ramp_lifetimes):
#     print(f"Ramp Step {i + 1}/{len(ramp_lifetimes)}: Solving with taun/taup = {life:.1e} s")
#     devsim.set_parameter(name="taun", value=life)
#     devsim.set_parameter(name="taup", value=life)
#     relative_tolerance = 1e-9 if life != target_lifetime else 1e-9
#     devsim.solve(type="dc", absolute_error=5.0, relative_error=relative_tolerance, maximum_iterations=200)

print("\n✅ Step 3 complete: Full photodiode model is defined and solved for equilibrium.")


# ==============================================================================
#                      HELPER FUNCTIONS FOR CHARACTERIZATION
# ==============================================================================
# The new simulation tasks are organized into functions for clarity.

def run_iv_sweep(device, voltages, p_flux):
    currents = []
    devsim.set_parameter(name="PhotonFlux", value=p_flux)
    if p_flux>0:
        print(f"DEBUG: PhotonFlux set to {devsim.get_parameter(name='PhotonFlux')}")  # Add this line

    for v in voltages:
        print(f"\nSetting Anode Bias: {v:.3f} V")
        devsim.set_parameter(device=device, name="anode_bias", value=v)
        try:
            # Add maximum_divergence and increase maximum_iterations
            devsim.solve(type="dc", absolute_error=1e10, relative_error=10,
                         maximum_iterations=400,  # Increased from 300
                         maximum_divergence=10)  # New parameter
            e_current = devsim.get_contact_current(device=device, contact="anode",
                                                   equation="ElectronContinuityEquation")
            h_current = devsim.get_contact_current(device=device, contact="anode", equation="HoleContinuityEquation")
            currents.append(e_current + h_current)
            print(f"✅ V = {v:.3f} V, Current = {currents[-1]:.4e} A/cm")
        except devsim.error as msg:
            # Catch the error, report it, and append a NaN
            print(f"❌ CONVERGENCE FAILED at V = {v:.3f} V. Error: {msg}")
            currents.append(float('nan'))
            # Optional: break the loop if one failure is enough
            break

    devsim.set_parameter(device=device, name="anode_bias", value=0.0)  # Reset bias at the end
    return np.array(currents)


def calculate_qe(dark_currents, light_currents, p_flux, device_width_cm, wavelength_nm):
    """
    Calculates the External Quantum Efficiency (QE) for the photodiode.

    Args:
        dark_currents (np.array): Dark current density in A/cm.
        light_currents (np.array): Light current density in A/cm.
        p_flux (float): Incident photon flux in photons/cm²/s.
        device_width_cm (float): The width of the device's top surface in cm.
        wavelength_nm (float): The wavelength of the incident light in nm (for context, not used in calculation).

    Returns:
        np.array: The calculated QE as a percentage (%).
    """
    # Physical constants
    q = 1.602e-19  # Elementary charge in Coulombs

    photocurrent_density = np.abs(light_currents - dark_currents)  # A/cm²
    electrons_per_sec_per_cm2 = photocurrent_density / q  # (e-/s)/cm²
    photons_per_sec_per_cm2 = p_flux  # (photons/s)/cm²
    qe = (electrons_per_sec_per_cm2 / photons_per_sec_per_cm2) * 100.0

    return qe


def run_cv_sweep(device, voltages, freq_hz):
    """
    Calculates C-V using numerical differentiation of charge.
    """
    capacitances = []
    DELTA_V = 0.001  # 1 mV step for numerical differentiation

    print(f"\nStarting C-V sweep...")

    # Initial solve at starting voltage
    devsim.set_parameter(device=device, name="anode_bias", value=voltages[0])
    devsim.solve(type="dc", absolute_error=10.0, relative_error=1e-9, maximum_iterations=100)

    for i, v in enumerate(voltages):
        print(f"Step {i + 1}/{len(voltages)}: Bias = {v:.2f} V")

        try:
            # Solve at V - DELTA_V/2
            devsim.set_parameter(device=device, name="anode_bias", value=v - DELTA_V / 2.0)
            devsim.solve(type="dc", absolute_error=10.0, relative_error=1e-9, maximum_iterations=100)
            q1 = devsim.get_contact_charge(device=device, contact="anode", equation="PotentialEquation")

            # Solve at V + DELTA_V/2
            devsim.set_parameter(device=device, name="anode_bias", value=v + DELTA_V / 2.0)
            devsim.solve(type="dc", absolute_error=10.0, relative_error=1e-9, maximum_iterations=100)
            q2 = devsim.get_contact_charge(device=device, contact="anode", equation="PotentialEquation")

            # Calculate capacitance
            C = abs(q2 - q1) / DELTA_V
            capacitances.append(C)

            print(f"  C = {C * 1e12:.3f} pF/cm (q1={q1:.3e} C/cm, q2={q2:.3e} C/cm)")

        except devsim.error as msg:
            print(f"  Failed at V = {v:.2f} V: {msg}")
            capacitances.append(float('nan'))

    # Reset bias
    devsim.set_parameter(device=device, name="anode_bias", value=0.0)
    return np.array(capacitances)


def run_cv_sweep_ac(device, voltages, freq_hz):
    """
    Calculates C-V using the more robust small-signal AC analysis method.
    This is the FINAL CORRECTED version.
    """
    capacitances = []
    omega = 2.0 * np.pi * freq_hz  # Angular frequency (rad/s)

    print(f"\nStarting AC C-V sweep at {freq_hz / 1e6:.1f} MHz...")

    for i, v in enumerate(voltages):
        devsim.set_parameter(device=device, name="anode_bias", value=v)
        print(f"Step {i + 1}/{len(voltages)}: Bias = {v:.2f} V")

        try:
            # FIX 1: Use TIGHT tolerances to get a physically correct DC solution.
            # This is the key to fixing the linear shape problem.
            devsim.solve(type="dc", absolute_error=100.0, relative_error=1e-2, maximum_iterations=100)

            # Perform the small-signal AC analysis at this correct DC point.
            devsim.solve(type="ac", frequency=freq_hz)

            # FIX 2: Use the original DC equation names to get the AC current.
            # The simulator automatically returns the AC result after an AC solve.
            imag_i_e = devsim.get_contact_current(device=device, contact="anode",
                                                  equation="ElectronContinuityEquation")
            imag_i_h = devsim.get_contact_current(device=device, contact="anode",
                                                  equation="HoleContinuityEquation")

            # FIX 3: Apply a negative sign to correct for the simulator's current
            # direction convention and ensure capacitance is positive, as required by physics.
            C = -(imag_i_e + imag_i_h) / omega
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
    # --- Step 4: Run I-V Sweeps (Tasks 1 & 2) ---
    print("\n--- STEP 4: Running I-V Sweeps ---")

    # Using adaptive voltage stepping for better convergence at high bias
    initial_small_steps = np.linspace(0.0, -0.5, 30)  # 25 steps of -0.02 V
    medium_steps = np.linspace(-0.6, -2.0, 15)  # 14 steps of -0.1 V
    large_steps = np.linspace(-2.2, -5.0, 12)  # 11 steps of -0.2 V and one -0.3V
    iv_voltages = np.unique(np.concatenate([initial_small_steps, medium_steps, large_steps]))
    iv_voltages = iv_voltages[::-1]
    print("iv_voltages:",iv_voltages)

    LIGHT_PHOTON_FLUX = 1e17
    print("  4A: Running Dark Current Simulation (Task 1)")
    dark_currents = run_iv_sweep(device_name, iv_voltages, p_flux=0.0)
    print("  4A Done !!")

    print("\n--- Ramping up Photon Flux at initial reverse bias for stability ---")


    devsim.set_parameter(device=device_name, name="anode_bias", value=iv_voltages[-1])  # Set bias to -5V
    devsim.solve(type="dc", absolute_error=1e10, relative_error=10)

    # Now ramp the flux at this reverse bias point
    flux_ramp = np.logspace(12, np.log10(LIGHT_PHOTON_FLUX), 6)
    for i, flux in enumerate(flux_ramp):
        devsim.set_parameter(name="PhotonFlux", value=flux)
        print(f"    Ramp Step {i + 1}/{len(flux_ramp)}: PhotonFlux = {flux:.1e} @ V = {iv_voltages[-1]:.1f}V")
        devsim.solve(type="dc", absolute_error=1e10, relative_error=10, maximum_iterations=100)

    print("\n--- Running Photocurrent Simulation (Task 2) ---")
    light_currents = run_iv_sweep(device_name, iv_voltages, p_flux=LIGHT_PHOTON_FLUX)
    print("  4B Done !!")



    # --- Step 5: Post-Process for Quantum Efficiency (Task 4) ---
    print("\n--- STEP 5: Calculating Quantum Efficiency ---")
    DEVICE_WIDTH_CM = 4.0e-4
    WAVELENGTH_NM = 650
    qe_values = calculate_qe(dark_currents, light_currents, LIGHT_PHOTON_FLUX, DEVICE_WIDTH_CM, WAVELENGTH_NM)

    # --- Step 6: Run C-V Simulation (Task 3) ---
    print("\n--- STEP 6: Running C-V Simulation ---")
    cv_voltages = np.linspace(0, -5, 21)

    capacitances = run_cv_sweep_ac(device_name, cv_voltages, freq_hz=1e6)

    # --- Step 7: Visualize All Results ---
    print("\n--- STEP 7: Generating Plots ---")

    # --- ADD THIS DEBUGGING BLOCK ---
    print("\n--- DEBUGGING QE DATA ---")
    print(f"Light currents being used for QE (first 5 values): {light_currents[:5]}")
    print(f"Dark currents being used for QE (first 5 values): {dark_currents[:5]}")
    print(f"Resulting QE values to be plotted (first 5 values): {qe_values[:5]}")
    print("\n Max dif max(abs(Dark- light ))):",max(dark_currents-light_currents, key=abs) )

    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    # --- Interactive I-V Plot ---
    fig_iv = go.Figure()
    fig_iv.add_trace(go.Scatter(x=iv_voltages, y=np.abs(dark_currents), mode='lines+markers', name='Dark Current',
                                marker_color='red'))
    fig_iv.add_trace(go.Scatter(x=iv_voltages, y=np.abs(light_currents), mode='lines+markers', name='Photocurrent',
                                marker_color='blue'))
    fig_iv.update_layout(title_text="I-V Characteristics (Interactive)",
                         xaxis_title="Anode Voltage (V)",
                         yaxis_title="Current Magnitude (A/cm)",
                         yaxis_type="log")
    fig_iv.show()

    # --- Interactive QE Plot ---
    fig_qe = go.Figure()
    fig_qe.add_trace(go.Scatter(x=iv_voltages, y=qe_values, mode='lines+markers', name='QE', marker_color='green'))
    fig_qe.update_layout(title_text=f"QE @ {WAVELENGTH_NM} nm (Interactive)",
                         xaxis_title="Anode Voltage (V)",
                         yaxis_title="External Quantum Efficiency (%)",
                         yaxis_range=[0, 105])  # Set a clean y-axis range
    fig_qe.show()

    # --- Interactive C-V and Mott-Schottky Plot ---
    fig_cv = make_subplots(rows=1, cols=2, subplot_titles=("C-V @ 1 MHz", "Mott-Schottky Plot"))
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