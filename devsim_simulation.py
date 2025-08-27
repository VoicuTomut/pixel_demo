# devsim_simulation.py
# Steps 1, 2, & 3 (Corrected with all verification steps)

import devsim
import os

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
alpha = 1e4        # Units: 1/cm (e.g., for visible light)


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
devsim.add_gmsh_interface(mesh=device_name, gmsh_name="pn_junction", region0="p_region", region1="n_region", name="pn_junction")
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
def set_silicon_parameters(device, region):
    """Sets the basic material parameters for Silicon."""
    devsim.set_parameter(device=device, region=region, name="Permittivity", value=11.9 * 8.854e-14)
    devsim.set_parameter(device=device, region=region, name="IntrinsicCarrierDensity", value=1.0e10)
    devsim.set_parameter(device=device, region=region, name="ElectronCharge", value=1.6e-19)
    devsim.set_parameter(device=device, region=region, name="taun", value=1.0e-7)
    devsim.set_parameter(device=device, region=region, name="taup", value=1.0e-7)

def define_doping(device, p_doping, n_doping):
    """Defines the acceptor and donor concentrations for the device."""
    devsim.node_model(device=device, region="p_region", name="Acceptors", equation=f"{p_doping}")
    devsim.node_model(device=device, region="p_region", name="Donors",    equation="0.0")
    devsim.node_model(device=device, region="n_region", name="Acceptors", equation="0.0")
    devsim.node_model(device=device, region="n_region", name="Donors",    equation=f"{n_doping}")
    for region in ["p_region", "n_region"]:
        devsim.node_model(device=device, region=region, name="NetDoping", equation="Donors - Acceptors")
    print(f"Defined doping: N_A = {p_doping:.1e} cm^-3, N_D = {n_doping:.1e} cm^-3")


def define_mobility_models(device, region):
    """
    Defines doping-dependent mobility using the Caughey-Thomas model.
    Parameters are from Sze, "Physics of Semiconductor Devices", Ch. 1, Table 2, p. 42.
    """
    # Total impurity concentration is the sum of donors and acceptors
    devsim.node_model(device=device, region=region, name="TotalDoping", equation="abs(Acceptors) + abs(Donors)")

    # Parameters for Electron Mobility Model
    mu_max_n = 1417.0
    mu_min_n = 68.5
    N_ref_n = 1.10e17
    alpha_n = 0.711

    # Caughey-Thomas Equation for Electrons
    eqn_n = f"{mu_min_n} + ({mu_max_n} - {mu_min_n}) / (1 + (TotalDoping / {N_ref_n})^{alpha_n})"
    devsim.node_model(device=device, region=region, name="ElectronMobility", equation=eqn_n)

    # Parameters for Hole Mobility Model
    mu_max_p = 470.5
    mu_min_p = 44.9
    N_ref_p = 2.23e17
    alpha_p = 0.719

    # Caughey-Thomas Equation for Holes
    eqn_p = f"{mu_min_p} + ({mu_max_p} - {mu_min_p}) / (1 + (TotalDoping / {N_ref_p})^{alpha_p})"
    devsim.node_model(device=device, region=region, name="HoleMobility", equation=eqn_p)


# --- Execute Step 2 ---
print("\nSetting silicon material parameters...")
set_silicon_parameters(device=device_name, region="p_region")
set_silicon_parameters(device=device_name, region="n_region")

define_doping(device=device_name, p_doping=1e16, n_doping=1e18)

print("Defining doping-dependent mobility models...")
define_mobility_models(device=device_name, region="p_region")
define_mobility_models(device=device_name, region="n_region")

print("\n--- Step 2 complete: Physics and doping defined ---")

# ==============================================================================
# STEP 3: SETTING UP THE PHOTODIODE PHYSICAL MODEL AND EQUATIONS
# ==============================================================================
print("--- STEP 3: Setting Up Full Photodiode Physical Model ---")

# --- Part A: Create Solution Variables ---
print("  3A: Creating solution variables (Potential, Electrons, Holes)...")
# 📚 Sze Ref: These are the core variables (ψ, n, p) from Ch. 2, Sec. 2.2, p. 81.
for region in ["p_region", "n_region"]:
    devsim.node_solution(device=device_name, region=region, name="Potential")
    devsim.node_solution(device=device_name, region=region, name="Electrons")
    devsim.node_solution(device=device_name, region=region, name="Holes")

# --- Part B: Define ALL Bulk Physical Models ---
print("  3B: Defining all bulk physical models...")

# Set global parameters needed by the models.
# Thermal Voltage (Vt = kT/q) is a fundamental constant in carrier statistics and transport.
devsim.set_parameter(name="ThermalVoltage", value=0.0259)  # Value at 300K
# Set the optical simulation parameters from the variables defined at the top of the script.
devsim.set_parameter(name="PhotonFlux", value=photon_flux)
devsim.set_parameter(name="alpha", value=alpha)

# Loop through both semiconductor regions to define the models in each.
for region in ["p_region", "n_region"]:

    # --- Electrostatics Models ---
    # These models describe the relationship between voltage, electric field, and charge.

    # Make the 'Potential' solution available on the edges between nodes.
    # This creates the variables Potential@n0 and Potential@n1 for use in edge models.
    devsim.edge_from_node_model(device=device_name, region=region, node_model="Potential")

    # PURPOSE: Models the Electric Field (E-Field) along each edge of the mesh.
    # 📚 Sze Ref: Chapter 2, "p-n Junctions", Eq. 2, p. 81.
    # FORMULA: E = -∇ψ  (approximated as E ≈ -(ψ₁ - ψ₀)/L )
    devsim.edge_model(device=device_name, region=region, name="ElectricField",
                      equation="(Potential@n0 - Potential@n1) * EdgeInverseLength")

    # PURPOSE: Models the Electric Displacement Field (D-Field). This is the flux term in Poisson's equation.
    # FORMULA: D = ε * E, where ε is the material permittivity.
    devsim.edge_model(device=device_name, region=region, name="DField",
                      equation="Permittivity * ElectricField")
    # The solver requires the derivatives of the D-Field with respect to the Potential at each node.
    devsim.edge_model(device=device_name, region=region, name="DField:Potential@n0",
                      equation="Permittivity * EdgeInverseLength")
    devsim.edge_model(device=device_name, region=region, name="DField:Potential@n1",
                      equation="-Permittivity * EdgeInverseLength")

    # PURPOSE: Models the total space charge density (ρ) at each node.
    # 📚 Sze Ref: Chapter 2, "p-n Junctions", Eq. 1, p. 81.
    # FORMULA: ρ = q * (p - n + N_d⁺ - N_a⁻)
    devsim.node_model(device=device_name, region=region, name="SpaceCharge",
                      equation="ElectronCharge * (Holes - Electrons + NetDoping)")

    # --- Carrier and Current Transport Models ---
    # These models describe how electrons and holes move through the device.

    # Helper model for the potential difference between nodes, normalized by the thermal voltage.
    # This term (vdiff) is the argument for the Bernoulli functions used in the current equations.
    devsim.edge_model(device=device_name, region=region, name="vdiff",
                      equation="(Potential@n0 - Potential@n1)/ThermalVoltage")
    # The Bernoulli function is central to the Scharfetter-Gummel formulation for current density.
    # DEVSIM has a built-in function B(x) = x / (exp(x) - 1).
    devsim.edge_model(device=device_name, region=region, name="Bernoulli_vdiff", equation="B(vdiff)")
    devsim.edge_model(device=device_name, region=region, name="Bernoulli_neg_vdiff", equation="B(-vdiff)")

    # Make the carrier concentrations available on the edges.
    devsim.edge_from_node_model(device=device_name, region=region, node_model="Electrons")
    devsim.edge_from_node_model(device=device_name, region=region, node_model="Holes")

    # Explicitly average mobility onto the edge for stability.
    # Using a node model (like ElectronMobility) directly in an edge model can cause asymmetry
    # in the solver. A symmetric average is more robust and physically correct.
    devsim.edge_from_node_model(device=device_name, region=region, node_model="ElectronMobility")
    devsim.edge_from_node_model(device=device_name, region=region, node_model="HoleMobility")
    devsim.edge_model(device=device_name, region=region, name="EdgeElectronMobility",
                      equation="(ElectronMobility@n0 + ElectronMobility@n1) * 0.5")
    devsim.edge_model(device=device_name, region=region, name="EdgeHoleMobility",
                      equation="(HoleMobility@n0 + HoleMobility@n1) * 0.5")

    # PURPOSE: Models the electron current density (Jn) using the Scharfetter-Gummel formulation.
    # 📚 Sze Ref: Chapter 2, "p-n Junctions", Eq. 4, p. 81.
    # FORMULA: Jn = q*μn*n*E + q*Dn*∇n  (Drift + Diffusion)
    electron_current_eq = "ElectronCharge * EdgeElectronMobility * ThermalVoltage * EdgeInverseLength * (Electrons@n1 * Bernoulli_neg_vdiff - Electrons@n0 * Bernoulli_vdiff)"
    devsim.edge_model(device=device_name, region=region, name="ElectronCurrent", equation=electron_current_eq)
    # Define all required derivatives for the solver using DEVSIM's symbolic differentiator.
    for v in ["Potential", "Electrons", "Holes"]:
        for n in ["n0", "n1"]:
            devsim.edge_model(device=device_name, region=region, name=f"ElectronCurrent:{v}@{n}",
                              equation=f"diff({electron_current_eq}, {v}@{n})")

    # PURPOSE: Models the hole current density (Jp).
    # 📚 Sze Ref: Chapter 2, "p-n Junctions", Eq. 5, p. 81.
    # FORMULA: Jp = q*μp*p*E - q*Dp*∇p  (Drift + Diffusion)
    hole_current_eq = "-ElectronCharge * EdgeHoleMobility * ThermalVoltage * EdgeInverseLength * (Holes@n1 * Bernoulli_vdiff - Holes@n0 * Bernoulli_neg_vdiff)"
    devsim.edge_model(device=device_name, region=region, name="HoleCurrent", equation=hole_current_eq)
    for v in ["Potential", "Electrons", "Holes"]:
        for n in ["n0", "n1"]:
            devsim.edge_model(device=device_name, region=region, name=f"HoleCurrent:{v}@{n}",
                              equation=f"diff({hole_current_eq}, {v}@{n})")

    # PURPOSE: A helper model for -Jp, required for the Hole Continuity Equation.
    devsim.edge_model(device=device_name, region=region, name="NegHoleCurrent", equation="-HoleCurrent")
    for v in ["Potential", "Electrons", "Holes"]:
        for n in ["n0", "n1"]:
            devsim.edge_model(device=device_name, region=region, name=f"NegHoleCurrent:{v}@{n}",
                              equation=f"-HoleCurrent:{v}@{n}")

    # --- Recombination and Generation Models ---
    # These models describe the creation and annihilation of electron-hole pairs.

    # Helper models for calculating equilibrium carrier concentrations (n₀ and p₀), needed for the SRH model.
    devsim.node_model(device=device_name, region=region, name="n_i_squared",
                      equation="IntrinsicCarrierDensity^2")
    devsim.node_model(device=device_name, region=region, name="IntrinsicElectrons",
                      equation="0.5*(NetDoping+(NetDoping^2+4*n_i_squared)^0.5)")
    devsim.node_model(device=device_name, region=region, name="IntrinsicHoles",
                      equation="0.5*(-NetDoping+(NetDoping^2+4*n_i_squared)^0.5)")

    # PURPOSE: Models Shockley-Read-Hall (SRH) recombination, the dominant thermal recombination process.
    # 📚 Sze Ref: Chapter 1, "Physics and Properties of Semiconductors", Eq. 52, p. 31.
    # FORMULA: U_srh = (p*n - n_i²) / [τ_p*(n + n₁) + τ_n*(p + p₁)]
    srh_eq = "(Electrons*Holes - n_i_squared) / (taup*(Electrons + IntrinsicElectrons) + taun*(Holes + IntrinsicHoles))"
    devsim.node_model(device=device_name, region=region, name="USRH", equation=srh_eq)

    # PURPOSE: Models the generation of electron-hole pairs due to light absorption.
    # 📚 Sze Ref: Chapter 13, "Photodetectors and Solar Cells", p. 636.
    # FORMULA: G_opt(d) = Φ * α * exp(-α*d), where 'd' is the depth from the surface.
    # IMPLEMENTATION: The top surface is at y=0, so the depth 'd' is (0.0 - y).
    devsim.node_model(device=device_name, region=region, name="OpticalGeneration",
                      equation="PhotonFlux * alpha * exp(-alpha * (0.0 - y))")

    # PURPOSE: Models the net rate of carrier change. This is the term that goes into the continuity equations.
    # FORMULA: U_net = U_srh - G_opt
    devsim.node_model(device=device_name, region=region, name="NetRecombination",
                      equation="USRH - OpticalGeneration")
    # Define derivatives for the solver. G_opt is constant with respect to n and p.
    devsim.node_model(device=device_name, region=region, name="NetRecombination:Electrons",
                      equation=f"diff({srh_eq}, Electrons)")
    devsim.node_model(device=device_name, region=region, name="NetRecombination:Holes",
                      equation=f"diff({srh_eq}, Holes)")

# --- Part C: Define ALL Boundary Condition Models ---
print("  3C: Defining all boundary condition models...")
for contact in ["anode", "cathode"]:
    # CONDITION: Potential at contact = applied bias (ψ = V_applied)
    devsim.set_parameter(device=device_name, name=f"{contact}_bias", value=0.0)
    devsim.contact_node_model(device=device_name, contact=contact, name="contact_potential",
                              equation=f"Potential - {contact}_bias")
    devsim.contact_node_model(device=device_name, contact=contact, name="contact_potential:Potential", equation="1.0")
    # CONDITION: Carriers are in equilibrium at contact (p*n = ni^2)
    devsim.contact_node_model(device=device_name, contact=contact, name="contact_equilibrium",
                              equation="Electrons * Holes - n_i_squared")
    devsim.contact_node_model(device=device_name, contact=contact, name="contact_equilibrium:Electrons",
                              equation="Holes")
    devsim.contact_node_model(device=device_name, contact=contact, name="contact_equilibrium:Holes",
                              equation="Electrons")

    #  Define a model for charge neutrality at the contact
    # FORMULA: p - n + NetDoping = 0
    devsim.contact_node_model(device=device_name, contact=contact, name="contact_charge_neutrality",
                              equation="Holes - Electrons + NetDoping")
    devsim.contact_node_model(device=device_name, contact=contact, name="contact_charge_neutrality:Electrons",
                              equation="-1.0")
    devsim.contact_node_model(device=device_name, contact=contact, name="contact_charge_neutrality:Holes",
                              equation="1.0")


# CONDITION: ψ, n, and p are continuous across the p-n junction
for variable in ["Potential", "Electrons", "Holes"]:
    devsim.interface_model(device=device_name, interface="pn_junction", name=f"{variable}_continuity",
                           equation=f"{variable}@r0 - {variable}@r1")
    devsim.interface_model(device=device_name, interface="pn_junction", name=f"{variable}_continuity:{variable}@r0",
                           equation="1.0")
    devsim.interface_model(device=device_name, interface="pn_junction", name=f"{variable}_continuity:{variable}@r1",
                           equation="-1.0")


# --- Part D: Solve for Initial Equilibrium (Staged Method with Correct Initial Guess) ---
print("  3D: Solving for initial equilibrium state (two-step method)...")

# --- Create a physically-based initial guess for ALL variables ---
print("    Creating robust initial guess based on charge neutrality...")
for region in ["p_region", "n_region"]:
    # Set initial carriers based on doping
    devsim.set_node_values(device=device_name, region=region, name="Electrons", init_from="IntrinsicElectrons")
    devsim.set_node_values(device=device_name, region=region, name="Holes", init_from="IntrinsicHoles")

    # CORRECTED: Set initial potential based on carrier concentrations using the Boltzmann relation.
    # This correctly calculates potential in Volts and creates a physically consistent starting point.
    # 📚 Sze Ref: Based on Ch. 2, Sec. 2.2, Eqs. 10 & 11, p. 82, relating potential to carrier levels.
    # FORMULA: ψ = Vt * log(n/ni)
    devsim.node_model(device=device_name, region=region, name="InitialPotential",
                      equation="ThermalVoltage * log(IntrinsicElectrons/IntrinsicCarrierDensity)")
    devsim.set_node_values(device=device_name, region=region, name="Potential", init_from="InitialPotential")


# --- First Solve: Potential Only ---
print("    Step 1/2: Assembling and solving for Potential only...")
# In this step, we ONLY assemble the Potential equation and its boundary conditions.
for region in ["p_region", "n_region"]:
    # 📚 Sze Ref: Ch. 2, Eq. 1, p. 81.
    # FORMULA: Poisson's Equation: ∇²ψ = -ρ/ε
    devsim.equation(device=device_name, region=region, name="PotentialEquation", variable_name="Potential",
                    node_model="SpaceCharge", edge_model="DField", variable_update="log_damp")
for contact in ["anode", "cathode"]:
    # Boundary condition for Potential at contacts
    devsim.contact_equation(device=device_name, contact=contact, name="PotentialEquation",
                            node_model="contact_potential")
# Continuity condition for Potential at the interface
devsim.interface_equation(device=device_name, interface="pn_junction", name="PotentialEquation",
                          interface_model="Potential_continuity", type="continuous")

# Solve for Potential with a tight tolerance to get an accurate starting point
devsim.solve(type="dc", absolute_error=1e-10, relative_error=1e-12, maximum_iterations=50)


# --- Update carrier guess based on the newly solved potential ---
print("    Updating carrier guess using Boltzmann statistics...")
for region in ["p_region", "n_region"]:
    # This step makes the carrier distribution consistent with the solved potential profile.
    # FORMULAS: n ≈ ni*exp(ψ/Vt), p ≈ ni*exp(-ψ/Vt)
    devsim.node_model(device=device_name, region=region, name="UpdatedElectrons",
                      equation="IntrinsicCarrierDensity*exp(Potential/ThermalVoltage)")
    devsim.node_model(device=device_name, region=region, name="UpdatedHoles",
                      equation="IntrinsicCarrierDensity*exp(-Potential/ThermalVoltage)")
    devsim.set_node_values(device=device_name, region=region, name="Electrons", init_from="UpdatedElectrons")
    devsim.set_node_values(device=device_name, region=region, name="Holes", init_from="UpdatedHoles")


# --- Second Solve: Fully Coupled System ---
print("    Step 2/2: Assembling continuity equations and solving the full system...")
# Now, we assemble the carrier continuity equations and their boundary conditions.
for region in ["p_region", "n_region"]:
    # 📚 Sze Ref: Ch. 2, Eq. 6, p. 81.
    # FORMULA: Electron Continuity: (1/q)∇⋅Jn - (U-G) = 0
    devsim.equation(device=device_name, region=region, name="ElectronContinuityEquation", variable_name="Electrons",
                    node_model="NetRecombination", edge_model="ElectronCurrent", variable_update="log_damp") # Use log_damp for stability
    # 📚 Sze Ref: Ch. 2, Eq. 7, p. 81.
    # FORMULA: Hole Continuity: -(1/q)∇⋅Jp - (U-G) = 0
    devsim.equation(device=device_name, region=region, name="HoleContinuityEquation", variable_name="Holes",
                    node_model="NetRecombination", edge_model="NegHoleCurrent", variable_update="log_damp") # Use log_damp for stability

# Apply the independent boundary conditions for the carrier equations at the contacts
for contact in ["anode", "cathode"]:
    # Condition 1: Charge Neutrality (p - n + N_net = 0)
    devsim.contact_equation(device=device_name, contact=contact, name="ElectronContinuityEquation",
                            node_model="contact_charge_neutrality")
    # Condition 2: Mass-Action Law (p*n = ni²)
    devsim.contact_equation(device=device_name, contact=contact, name="HoleContinuityEquation",
                            node_model="contact_equilibrium")

# Continuity conditions for carriers at the interface
devsim.interface_equation(device=device_name, interface="pn_junction", name="ElectronContinuityEquation",
                          interface_model="Electrons_continuity", type="continuous")
devsim.interface_equation(device=device_name, interface="pn_junction", name="HoleContinuityEquation",
                          interface_model="Holes_continuity", type="continuous")

# Solve the complete, fully coupled system with more iterations
# --- Final Solve using a Ramping Strategy ---
print("\n--- Final Solve: Using a Ramping Strategy for Stability ---")
print("The simulation is numerically stiff. Ramping carrier lifetimes to guide the solver.")

# Get the final, desired lifetime you set at the top of the script
try:
    target_lifetime = devsim.get_parameter(name="taun")
except devsim.error:
    # Set a default if it's not already a parameter (it should be from your setup)
    target_lifetime = 1e-7

# A list of lifetime values to ramp through.
# We start with a very short, numerically stable lifetime and end with the target value.
ramp_lifetimes = [1e-13, 1e-11, 1e-9, target_lifetime]

# Loop through the ramp values, using each solution as the guess for the next
for i, life in enumerate(ramp_lifetimes):
    print(f"Ramp Step {i+1}/{len(ramp_lifetimes)}: Solving with taun/taup = {life:.1e} s")

    # Update the physics model parameters for this step
    devsim.set_parameter(name="taun", value=life)
    devsim.set_parameter(name="taup", value=life)

    # For the final ramp step, use the original strict tolerance.
    # For intermediate steps, a looser tolerance is fine and faster.
    if life == target_lifetime:
        print("Final step: applying strict tolerance.")
        relative_tolerance = 1e-10
    else:
        relative_tolerance = 1e-6

    try:
        # Solve the system with the current lifetime value
        devsim.solve(type="dc", absolute_error=1e10, relative_error=relative_tolerance, maximum_iterations=100)
    except devsim.error as msg:
        print(f"\nConvergence failed during ramping at lifetime {life:.1e}.")
        print(f"Error: {msg}")
        # Stop the script if an intermediate step fails
        raise RuntimeError("Ramping failed to converge.")

# The final print statement from your script
print("\n✅ Step 3 complete: Full photodiode model is defined and solved for equilibrium.")
