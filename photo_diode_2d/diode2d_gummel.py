"""
2D Photodiode DEVSIM Simulation Setup - Enhanced & Corrected Version
-- MODIFIED TO USE GUMMEL METHOD FOR EQUILIBRIUM SOLVE --
"""

import devsim as ds
import math
import sys
import os
import matplotlib.pyplot as plt
import numpy as np


def add_poisson_physics(device_name):
    """
    STEP 1A: DEFINE VARIABLES & POISSON EQUATION ONLY

    Sets up only the electrostatic potential equation.
    This allows solving for potential first before adding carrier transport.
    """

    regions = ds.get_region_list(device=device_name)
    contacts = ds.get_contact_list(device=device_name)
    interfaces = ds.get_interface_list(device=device_name)

    print("\n" + "=" * 70)
    print("STEP 1A: POISSON-ONLY PHYSICS SETUP")
    print("=" * 70)
    print(f"\nConfiguring Poisson equation for {len(regions)} regions: {regions}")

    for region in regions:
        print(f"\n{'─' * 70}")
        print(f"Region: {region}")
        print(f"{'─' * 70}")

        # Step 1: Create solution variables (all three)
        create_solution_variables(device_name, region)

        # Step 2: Set initial conditions
        set_initial_conditions(device_name, region)

        # Step 3: Build electric field model (needed for Poisson)
        build_electric_field_model(device_name, region)

        # Step 4: Construct ONLY Poisson equation
        construct_poisson_eq(device_name, region)

    # Add contact boundary conditions for Poisson only
    print(f"\n{'─' * 70}")
    print("CONTACT BOUNDARY CONDITIONS (Poisson Only)")
    print(f"{'─' * 70}")

    for contact in contacts:
        if contact == 'cathode':
            region = 'n_plus_region'
            bias = 0.0
        elif contact == 'anode':
            region = 'p_plus_region'
            bias = 0.0
        else:
            print(f"Warning: Unknown contact {contact}, skipping...")
            continue

        add_poisson_contact(device_name, contact, region, bias)

    # Add interface conditions for Poisson only
    print(f"\n{'─' * 70}")
    print("INTERFACE CONDITIONS (Poisson Only)")
    print(f"{'─' * 70}")

    for interface in interfaces:
        add_poisson_interface(device_name, interface)

    print("\n" + "=" * 70)
    print("✓ POISSON PHYSICS SETUP COMPLETE")
    print("=" * 70)

    return device_name


def add_drift_diffusion_physics(device_name):
    """
    STEP 1B: ADD DRIFT-DIFFUSION EQUATIONS (Carrier Transport)

    Adds electron and hole continuity equations to the existing Poisson setup.
    """

    regions = ds.get_region_list(device=device_name)
    contacts = ds.get_contact_list(device=device_name)
    interfaces = ds.get_interface_list(device=device_name)

    print("\n" + "=" * 70)
    print("STEP 1B: DRIFT-DIFFUSION PHYSICS SETUP")
    print("=" * 70)
    print(f"\nAdding carrier transport equations for {len(regions)} regions: {regions}")

    for region in regions:
        print(f"\n{'─' * 70}")
        print(f"Region: {region}")
        print(f"{'─' * 70}")

        # Build carrier transport models
        build_drift_diffusion_model(device_name, region)
        build_recombination_model(device_name, region)

        # Construct carrier continuity equations
        construct_electron_continuity_eq(device_name, region)
        construct_hole_continuity_eq(device_name, region)


    # Update contact boundary conditions to include carriers
    print(f"\n{'─' * 70}")
    print("CONTACT BOUNDARY CONDITIONS (Adding Carrier BCs)")
    print(f"{'─' * 70}")

    for contact in contacts:
        if contact == 'cathode':
            region = 'n_plus_region'
        elif contact == 'anode':
            region = 'p_plus_region'
        else:
            print(f"Warning: Unknown contact {contact}, skipping...")
            continue

        add_carrier_contact_bc(device_name, contact, region)

    # Update interface conditions to include carriers
    print(f"\n{'─' * 70}")
    print("INTERFACE CONDITIONS (Adding Carrier Continuity)")
    print(f"{'─' * 70}")

    for interface in interfaces:
        add_carrier_interface(device_name, interface)

    print("\n" + "=" * 70)
    print("✓ DRIFT-DIFFUSION PHYSICS SETUP COMPLETE")
    print("=" * 70)

    return device_name


def add_poisson_contact(device_name, contact_name, region_name, bias=0.0):
    """Add contact boundary condition for Poisson equation only."""
    print(f"    Adding Poisson contact: {contact_name}")

    n_i = ds.get_parameter(device=device_name, name="n_i")
    N_D = ds.get_parameter(device=device_name, region=region_name, name="N_D")
    N_A = ds.get_parameter(device=device_name, region=region_name, name="N_A")
    V_t = ds.get_parameter(device=device_name, name="V_t")

    # Determine built-in potential
    if N_D > N_A:  # n-type contact
        V_bi = V_t * math.log(N_D / n_i)
    else:  # p-type contact
        V_bi = -V_t * math.log(N_A / n_i)

    ds.set_parameter(device=device_name, name=f"{contact_name}_bias", value=bias)

    # Contact potential BC
    ds.contact_node_model(
        device=device_name,
        contact=contact_name,
        name=f"{contact_name}_bc",
        equation=f"Potential - ({contact_name}_bias + {V_bi} )"
    )

    ds.contact_node_model(
        device=device_name,
        contact=contact_name,
        name=f"{contact_name}_bc:Potential",
        equation="1.0"
    )

    ds.contact_equation(
        device=device_name,
        contact=contact_name,
        name="PoissonEquation",
        node_model=f"{contact_name}_bc",
        edge_charge_model="ElectricFlux"
    )

    print(f"      ✓ Poisson contact {contact_name}: V={bias}V + V_bi={V_bi:.4f}V")


def add_carrier_contact_bc(device_name, contact_name, region_name):
    """Add carrier boundary conditions to an existing contact."""
    print(f"    Adding carrier BCs to contact: {contact_name}")

    n_i = ds.get_parameter(device=device_name, name="n_i")
    N_D = ds.get_parameter(device=device_name, region=region_name, name="N_D")
    N_A = ds.get_parameter(device=device_name, region=region_name, name="N_A")

    if N_D > N_A:  # n-type contact
        n_contact = N_D
        p_contact = n_i ** 2 / N_D
    else:  # p-type contact
        p_contact = N_A
        n_contact = n_i ** 2 / N_A

    # Electron BC
    ds.contact_node_model(
        device=device_name,
        contact=contact_name,
        name=f"{contact_name}_n_bc",
        equation=f"Electrons - {n_contact}"
    )

    ds.contact_node_model(
        device=device_name,
        contact=contact_name,
        name=f"{contact_name}_n_bc:Electrons",
        equation="1.0"
    )

    # Hole BC
    ds.contact_node_model(
        device=device_name,
        contact=contact_name,
        name=f"{contact_name}_p_bc",
        equation=f"Holes - {p_contact}"
    )

    ds.contact_node_model(
        device=device_name,
        contact=contact_name,
        name=f"{contact_name}_p_bc:Holes",
        equation="1.0"
    )

    ds.contact_equation(
        device=device_name,
        contact=contact_name,
        name="ElectronContinuity",
        node_model=f"{contact_name}_n_bc",
        edge_current_model="ElectronCurrent"
    )

    ds.contact_equation(
        device=device_name,
        contact=contact_name,
        name="HoleContinuity",
        node_model=f"{contact_name}_p_bc",
        edge_current_model="HoleCurrent"
    )

    print(f"      ✓ Carrier BCs: n={n_contact:.2e}, p={p_contact:.2e}")


def add_poisson_interface(device_name, interface_name):
    """Add interface condition for Poisson equation only."""
    print(f"    Adding Poisson interface: {interface_name}")

    ds.interface_model(
        device=device_name,
        interface=interface_name,
        name="continuous_potential",
        equation="Potential@r0 - Potential@r1"
    )

    ds.interface_model(
        device=device_name,
        interface=interface_name,
        name="continuous_potential:Potential@r0",
        equation="1.0"
    )

    ds.interface_model(
        device=device_name,
        interface=interface_name,
        name="continuous_potential:Potential@r1",
        equation="-1.0"
    )

    ds.interface_equation(
        device=device_name,
        interface=interface_name,
        name="PoissonEquation",
        interface_model="continuous_potential",
        type="continuous"
    )

    print(f"      ✓ Interface: continuous ψ")


def add_carrier_interface(device_name, interface_name):
    """Add carrier continuity conditions to an existing interface."""
    print(f"    Adding carrier interface conditions: {interface_name}")

    # Continuous electrons
    ds.interface_model(
        device=device_name,
        interface=interface_name,
        name="continuous_electrons",
        equation="Electrons@r0 - Electrons@r1"
    )

    ds.interface_model(
        device=device_name,
        interface=interface_name,
        name="continuous_electrons:Electrons@r0",
        equation="1.0"
    )

    ds.interface_model(
        device=device_name,
        interface=interface_name,
        name="continuous_electrons:Electrons@r1",
        equation="-1.0"
    )

    # Continuous holes
    ds.interface_model(
        device=device_name,
        interface=interface_name,
        name="continuous_holes",
        equation="Holes@r0 - Holes@r1"
    )

    ds.interface_model(
        device=device_name,
        interface=interface_name,
        name="continuous_holes:Holes@r0",
        equation="1.0"
    )

    ds.interface_model(
        device=device_name,
        interface=interface_name,
        name="continuous_holes:Holes@r1",
        equation="-1.0"
    )

    ds.interface_equation(
        device=device_name,
        interface=interface_name,
        name="ElectronContinuity",
        interface_model="continuous_electrons",
        type="continuous"
    )

    ds.interface_equation(
        device=device_name,
        interface=interface_name,
        name="HoleContinuity",
        interface_model="continuous_holes",
        type="continuous"
    )

    print(f"      ✓ Interface: continuous n, p")




def setup_photodiode_device(mesh_file):
    """
    Complete setup of photodiode device structure in DEVSIM.
    """
    device_name = "photodiode"
    print("\n" + "=" * 70)
    print("DEVSIM 2D PHOTODIODE DEVICE SETUP")
    print("=" * 70)
    q = 1.602e-19;
    k_B = 1.381e-23;
    eps_0 = 8.854e-14;
    T = 300.0
    V_t = k_B * T / q;
    eps_si = 11.7;
    n_i = 1.0e10;
    mu_n = 1400.0;
    mu_p = 450.0
    tau_n = 1.0e-6;
    tau_p = 1.0e-6;
    N_D_nplus = 1.0e19;
    N_A_p = 1.0e16
    N_A_pplus = 1.0e19
    print(f"\nPhysical Constants:\n  T: {T} K\n  V_t: {V_t:.4f} V\n  n_i: {n_i:.2e} cm^-3")

    print("\n" + "=" * 70)
    print("STEP 1: IMPORTING GMSH MESH")
    print("=" * 70)
    if not os.path.exists(mesh_file):
        raise FileNotFoundError(f"Mesh file not found: {mesh_file}")

    print(f"Reading mesh file: {mesh_file}")
    try:
        if device_name in ds.get_device_list(): ds.delete_device(device=device_name)
        if "diode_mesh" in ds.get_mesh_list(): ds.delete_mesh(mesh="diode_mesh")
        ds.create_gmsh_mesh(mesh="diode_mesh", file=mesh_file)
        ds.add_gmsh_region(mesh="diode_mesh", gmsh_name="n_plus_region", region="n_plus_region", material="Si")
        ds.add_gmsh_region(mesh="diode_mesh", gmsh_name="p_region", region="p_region", material="Si")
        ds.add_gmsh_region(mesh="diode_mesh", gmsh_name="p_plus_region", region="p_plus_region", material="Si")
        ds.add_gmsh_contact(mesh="diode_mesh", gmsh_name="cathode", region="n_plus_region", name="cathode",
                            material="metal")
        ds.add_gmsh_contact(mesh="diode_mesh", gmsh_name="anode", region="p_plus_region", name="anode",
                            material="metal")
        ds.add_gmsh_interface(mesh="diode_mesh", gmsh_name="pn_interface", region0="n_plus_region", region1="p_region",
                              name="pn_interface")
        ds.add_gmsh_interface(mesh="diode_mesh", gmsh_name="p_pplus_interface", region0="p_region",
                              region1="p_plus_region", name="p_pplus_interface")
        ds.finalize_mesh(mesh="diode_mesh")
        ds.create_device(mesh="diode_mesh", device=device_name)
    except Exception as e:
        raise RuntimeError(f"Failed to import mesh and create device: {e}")

    # ... (Verification steps from original script are kept, but omitted here for brevity)

    print("\n" + "=" * 70)
    print("STEP 4: SETTING GLOBAL PARAMETERS")
    print("=" * 70)
    global_params = {'T': T, 'q': q, 'k_B': k_B, 'eps_0': eps_0, 'V_t': V_t, 'n_i': n_i}
    for name, value in global_params.items():
        ds.set_parameter(device=device_name, name=name, value=value)
    print("✓ Global parameters set")

    print("\n" + "=" * 70)
    print("STEP 5: SETTING MATERIAL PARAMETERS AND DOPING")
    print("=" * 70)
    region_params = {
        'n_plus_region': {'N_D': N_D_nplus, 'N_A': 0.0},
        'p_region': {'N_D': 0.0, 'N_A': N_A_p},
        'p_plus_region': {'N_D': 0.0, 'N_A': N_A_pplus}
    }
    for region in ds.get_region_list(device=device_name):
        ds.set_parameter(device=device_name, region=region, name="eps_r", value=eps_si)
        ds.set_parameter(device=device_name, region=region, name="mu_n", value=mu_n)
        ds.set_parameter(device=device_name, region=region, name="mu_p", value=mu_p)
        ds.set_parameter(device=device_name, region=region, name="tau_n", value=tau_n)
        ds.set_parameter(device=device_name, region=region, name="tau_p", value=tau_p)
        ds.set_parameter(device=device_name, region=region, name="N_D", value=region_params[region]['N_D'])
        ds.set_parameter(device=device_name, region=region, name="N_A", value=region_params[region]['N_A'])
    print("\n✓ Material parameters set for all regions")

    return device_name


def create_solution_variables(device_name, region):
    print(f"  Creating solution variables in {region}...")
    ds.node_solution(device=device_name, region=region, name="Potential")
    ds.node_solution(device=device_name, region=region, name="Electrons")
    ds.node_solution(device=device_name, region=region, name="Holes")
    print(f"    ✓ Potential, Electrons, Holes created")


def set_initial_conditions(device_name, region):
    """
    MODIFIED: This function now also sets the initial guess for Potential
    based on the built-in potential of the region.
    """
    print(f"  Setting CLAMPED initial conditions in {region}...")
    num_nodes = len(ds.get_node_model_values(device=device_name, region=region, name="node_index"))

    # Get necessary parameters
    N_D = ds.get_parameter(device=device_name, region=region, name="N_D")
    N_A = ds.get_parameter(device=device_name, region=region, name="N_A")
    n_i = ds.get_parameter(device=device_name, name="n_i")
    V_t = ds.get_parameter(device=device_name, name="V_t")
    N_net = N_D - N_A

    # --- Carrier Initial Conditions (Unchanged) ---
    MIN_CARRIER = 1e8;
    MAX_CARRIER = 1e20
    if N_net > 0:
        n_init = max(MIN_CARRIER, min(MAX_CARRIER, N_net))
        p_init = max(MIN_CARRIER, min(MAX_CARRIER, n_i ** 2 / n_init))
    else:
        p_init = max(MIN_CARRIER, min(MAX_CARRIER, abs(N_net)))
        n_init = max(MIN_CARRIER, min(MAX_CARRIER, n_i ** 2 / p_init))

    ds.set_node_values(device=device_name, region=region, name="Electrons", values=[n_init] * num_nodes)
    ds.set_node_values(device=device_name, region=region, name="Holes", values=[p_init] * num_nodes)
    print(f"    ✓ CLAMPED: n={n_init:.2e}, p={p_init:.2e}")

    # --- POTENTIAL INITIAL CONDITION (NEW) ---
    # Calculate the built-in potential as the initial guess
    if N_net > 0:  # n-type region
        psi_init = V_t * math.log(N_D / n_i)
    else:  # p-type region
        psi_init = -V_t * math.log(N_A / n_i)

    # Set the initial potential across all nodes in the region
    ds.set_node_values(device=device_name, region=region, name="Potential", values=[psi_init] * num_nodes)
    print(f"    ✓ INITIALIZED: ψ={psi_init:.4f} V")


def build_electric_field_model(device_name, region):
    print(f"    Building electric field model...")
    ds.edge_from_node_model(device=device_name, region=region, node_model="Potential")
    ds.edge_model(device=device_name, region=region, name="ElectricField",
                  equation="(Potential@n0 - Potential@n1) * EdgeInverseLength")
    ds.edge_model(device=device_name, region=region, name="ElectricField:Potential@n0", equation="EdgeInverseLength")
    ds.edge_model(device=device_name, region=region, name="ElectricField:Potential@n1", equation="-EdgeInverseLength")
    print(f"      ✓ Electric field E = -∇ψ created")


def build_drift_diffusion_model(device_name, region):
    print(f"    Building drift-diffusion models...")
    mu_n = ds.get_parameter(device=device_name, region=region, name="mu_n")
    mu_p = ds.get_parameter(device=device_name, region=region, name="mu_p")
    q = ds.get_parameter(device=device_name, name="q")
    V_t = ds.get_parameter(device=device_name, name="V_t")
    ds.edge_model(device=device_name, region=region, name="vdiff", equation="(Potential@n0 - Potential@n1) / V_t")
    ds.edge_from_node_model(device=device_name, region=region, node_model="Electrons")
    ds.edge_from_node_model(device=device_name, region=region, node_model="Holes")
    ds.edge_model(device=device_name, region=region, name="ElectronCurrent",
                  equation=f"{q}*{mu_n}*V_t*EdgeInverseLength*(Electrons@n0*B(vdiff)-Electrons@n1*B(-vdiff))")
    ds.edge_model(device=device_name, region=region, name="ElectronCurrent:Potential@n0",
                  equation=f"{q}*{mu_n}*EdgeInverseLength*(Electrons@n0*dBdx(vdiff)+Electrons@n1*dBdx(-vdiff))/V_t")
    ds.edge_model(device=device_name, region=region, name="ElectronCurrent:Potential@n1",
                  equation=f"-{q}*{mu_n}*EdgeInverseLength*(Electrons@n0*dBdx(vdiff)+Electrons@n1*dBdx(-vdiff))/V_t")
    ds.edge_model(device=device_name, region=region, name="ElectronCurrent:Electrons@n0",
                  equation=f"{q}*{mu_n}*V_t*EdgeInverseLength*B(vdiff)")
    ds.edge_model(device=device_name, region=region, name="ElectronCurrent:Electrons@n1",
                  equation=f"-{q}*{mu_n}*V_t*EdgeInverseLength*B(-vdiff)")
    ds.edge_model(device=device_name, region=region, name="HoleCurrent",
                  equation=f"-{q}*{mu_p}*V_t*EdgeInverseLength*(Holes@n1*B(vdiff)-Holes@n0*B(-vdiff))")
    ds.edge_model(device=device_name, region=region, name="HoleCurrent:Potential@n0",
                  equation=f"-{q}*{mu_p}*EdgeInverseLength*(Holes@n1*dBdx(vdiff)+Holes@n0*dBdx(-vdiff))/V_t")
    ds.edge_model(device=device_name, region=region, name="HoleCurrent:Potential@n1",
                  equation=f"{q}*{mu_p}*EdgeInverseLength*(Holes@n1*dBdx(vdiff)+Holes@n0*dBdx(-vdiff))/V_t")
    ds.edge_model(device=device_name, region=region, name="HoleCurrent:Holes@n0",
                  equation=f"{q}*{mu_p}*V_t*EdgeInverseLength*B(-vdiff)")
    ds.edge_model(device=device_name, region=region, name="HoleCurrent:Holes@n1",
                  equation=f"-{q}*{mu_p}*V_t*EdgeInverseLength*B(vdiff)")
    print(f"      ✓ Currents with CORRECTED derivatives")


def build_recombination_model(device_name, region, srh=True, radiative=False, auger=False):
    print(f"    Building recombination models...")
    n_i = ds.get_parameter(device=device_name, name="n_i")
    tau_n = ds.get_parameter(device=device_name, region=region, name="tau_n")
    tau_p = ds.get_parameter(device=device_name, region=region, name="tau_p")
    n_i_sq = n_i ** 2
    U_total = "0";
    dU_n = "0";
    dU_p = "0"
    if srh:
        n1, p1 = n_i, n_i
        ds.node_model(device=device_name, region=region, name="U_SRH",
                      equation=f"(Electrons*Holes - {n_i_sq}) / ({tau_p}*(Electrons + {n1}) + {tau_n}*(Holes + {p1}))")
        ds.node_model(device=device_name, region=region, name="U_SRH:Electrons",
                      equation=f"(Holes*({tau_p}*(Electrons+{n1})+{tau_n}*(Holes+{p1})) - (Electrons*Holes-{n_i_sq})*{tau_p}) / (({tau_p}*(Electrons+{n1})+{tau_n}*(Holes+{p1}))^2)")
        ds.node_model(device=device_name, region=region, name="U_SRH:Holes",
                      equation=f"(Electrons*({tau_p}*(Electrons+{n1})+{tau_n}*(Holes+{p1})) - (Electrons*Holes-{n_i_sq})*{tau_n}) / (({tau_p}*(Electrons+{n1})+{tau_n}*(Holes+{p1}))^2)")
        ds.node_model(device=device_name, region=region, name="U_SRH:Potential", equation="0")
        U_total += " + U_SRH";
        dU_n += " + U_SRH:Electrons";
        dU_p += " + U_SRH:Holes"
        print(f"      ✓ SRH recombination added")
    ds.node_model(device=device_name, region=region, name="Recombination", equation=U_total)
    ds.node_model(device=device_name, region=region, name="Recombination:Electrons", equation=dU_n)
    ds.node_model(device=device_name, region=region, name="Recombination:Holes", equation=dU_p)
    ds.node_model(device=device_name, region=region, name="Recombination:Potential", equation="0")


def construct_poisson_eq(device_name, region):
    print(f"    Constructing Poisson's equation...")
    q = ds.get_parameter(device=device_name, name="q")
    eps_0 = ds.get_parameter(device=device_name, name="eps_0")
    eps_r = ds.get_parameter(device=device_name, region=region, name="eps_r")
    epsilon = eps_0 * eps_r
    N_D = ds.get_parameter(device=device_name, region=region, name="N_D")
    N_A = ds.get_parameter(device=device_name, region=region, name="N_A")
    ds.node_model(device=device_name, region=region, name="NetDoping", equation=f"{N_D} - {N_A}")
    ds.node_model(device=device_name, region=region, name="SpaceCharge",
                  equation=f"{q} * (Holes - Electrons + NetDoping)")
    ds.node_model(device=device_name, region=region, name="SpaceCharge:Electrons", equation=f"-{q}")
    ds.node_model(device=device_name, region=region, name="SpaceCharge:Holes", equation=f"{q}")
    ds.edge_model(device=device_name, region=region, name="ElectricFlux", equation=f"{epsilon} * ElectricField")
    ds.edge_model(device=device_name, region=region, name="ElectricFlux:Potential@n0",
                  equation=f"{epsilon} * ElectricField:Potential@n0")
    ds.edge_model(device=device_name, region=region, name="ElectricFlux:Potential@n1",
                  equation=f"{epsilon} * ElectricField:Potential@n1")
    ds.node_model(device=device_name, region=region, name="PoissonSource", equation="-SpaceCharge")
    ds.node_model(device=device_name, region=region, name="PoissonSource:Electrons", equation="-SpaceCharge:Electrons")
    ds.node_model(device=device_name, region=region, name="PoissonSource:Holes", equation="-SpaceCharge:Holes")
    ds.equation(device=device_name, region=region, name="PoissonEquation", variable_name="Potential",
                edge_model="ElectricFlux", node_model="PoissonSource", variable_update="log_damp")
    print(f"      ✓ Poisson equation: ∇·(ε∇ψ) - ρ = 0")


def construct_electron_continuity_eq(device_name, region):
    print(f"    Constructing electron continuity equation...")
    q = ds.get_parameter(device=device_name, name="q")
    ds.node_model(device=device_name, region=region, name="OpticalGeneration", equation="0.0")
    ds.node_model(device=device_name, region=region, name="ElectronGenerationSource",
                  equation=f"-{q} * (Recombination - OpticalGeneration)")
    ds.node_model(device=device_name, region=region, name="ElectronGenerationSource:Electrons",
                  equation=f"-{q} * Recombination:Electrons")
    ds.node_model(device=device_name, region=region, name="ElectronGenerationSource:Holes",
                  equation=f"-{q} * Recombination:Holes")
    ds.node_model(device=device_name, region=region, name="ElectronGenerationSource:Potential",
                  equation=f"-{q} * Recombination:Potential")
    ds.equation(device=device_name, region=region, name="ElectronContinuity", variable_name="Electrons",
                edge_model="ElectronCurrent", node_model="ElectronGenerationSource", variable_update="log_damp")
    print(f"      ✓ Electron continuity: ∇·J_n - q(U-G) = 0")


def construct_hole_continuity_eq(device_name, region):
    print(f"    Constructing hole continuity equation...")
    q = ds.get_parameter(device=device_name, name="q")
    ds.node_model(device=device_name, region=region, name="HoleGenerationSource",
                  equation=f"{q} * (Recombination - OpticalGeneration)")
    ds.node_model(device=device_name, region=region, name="HoleGenerationSource:Electrons",
                  equation=f"{q} * Recombination:Electrons")
    ds.node_model(device=device_name, region=region, name="HoleGenerationSource:Holes",
                  equation=f"{q} * Recombination:Holes")
    ds.node_model(device=device_name, region=region, name="HoleGenerationSource:Potential",
                  equation=f"{q} * Recombination:Potential")
    ds.equation(device=device_name, region=region, name="HoleContinuity", variable_name="Holes",
                edge_model="HoleCurrent", node_model="HoleGenerationSource", variable_update="log_damp")
    print(f"      ✓ Hole continuity: ∇·J_p + q(U-G) = 0")


def restore_poisson_system(device_name):
    """
    Lightweight function to restore ONLY the Poisson equation system.
    It assumes all underlying models already exist.
    """
    regions = ds.get_region_list(device=device_name)
    contacts = ds.get_contact_list(device=device_name)
    interfaces = ds.get_interface_list(device=device_name)

    # Restore in bulk regions
    for region in regions:
        ds.equation(device=device_name, region=region, name="PoissonEquation", variable_name="Potential",
                    edge_model="ElectricFlux", node_model="PoissonSource",
                    variable_update="log_damp")

    # Restore at contacts
    for contact in contacts:
        ds.contact_equation(device=device_name, contact=contact, name="PoissonEquation",
                            node_model=f"{contact}_bc", edge_charge_model="ElectricFlux")

    # Restore at interfaces
    for interface in interfaces:
        ds.interface_equation(device=device_name, interface=interface, name="PoissonEquation",
                              interface_model="continuous_potential", type="continuous")
    print("      - Poisson system restored with log_damp.") # Updated print statement


def restore_carrier_system(device_name):
    """
    Lightweight function to restore ONLY the carrier continuity equation system.
    It assumes all underlying models already exist.
    """
    regions = ds.get_region_list(device=device_name)
    contacts = ds.get_contact_list(device=device_name)
    interfaces = ds.get_interface_list(device=device_name)

    # Restore in bulk regions
    for region in regions:
        ds.equation(device=device_name, region=region, name="ElectronContinuity", variable_name="Electrons",
                    edge_model="ElectronCurrent", node_model="ElectronGenerationSource",
                    variable_update="log_damp")
        ds.equation(device=device_name, region=region, name="HoleContinuity", variable_name="Holes",
                    edge_model="HoleCurrent", node_model="HoleGenerationSource",
                    variable_update="log_damp")

    # Restore at contacts
    for contact in contacts:
        ds.contact_equation(device=device_name, contact=contact, name="ElectronContinuity",
                            node_model=f"{contact}_n_bc", edge_current_model="ElectronCurrent")
        ds.contact_equation(device=device_name, contact=contact, name="HoleContinuity",
                            node_model=f"{contact}_p_bc", edge_current_model="HoleCurrent")

    # Restore at interfaces
    for interface in interfaces:
        ds.interface_equation(device=device_name, interface=interface, name="ElectronContinuity",
                              interface_model="continuous_electrons", type="continuous")
        ds.interface_equation(device=device_name, interface=interface, name="HoleContinuity",
                              interface_model="continuous_holes", type="continuous")
    print("      - Carrier system restored with log_damp.") # Updated print statement


def solve_equilibrium_gummel(device_name):
    """
    Solves the fully coupled drift-diffusion system using a robust and EFFICIENT
    fully DECOUPLED GUMMEL METHOD.
    """
    print("\n" + "=" * 70)
    print("GUMMEL METHOD EQUILIBRIUM SOLVER (EFFICIENT & ROBUST)")
    print("=" * 70)

    # --- Gummel Iteration Parameters ---
    max_gummel_iterations = 100
    convergence_tolerance = 1e-5
    min_check_iter = 3

    regions = ds.get_region_list(device=device_name)
    contacts = ds.get_contact_list(device=device_name)
    interfaces = ds.get_interface_list(device=device_name)

    # --- The Gummel Loop ---
    for iteration in range(max_gummel_iterations):
        print(f"\n{'─' * 30} Gummel Iteration {iteration + 1} {'─' * 30}")
        poisson_error = -1.0
        carrier_error = -1.0

        try:
            # ======================================================================
            # STEP 1: Solve for Potential ONLY (n, p are frozen)
            # ======================================================================
            print("  [1/2] Solving for Potential (n, p frozen)...")
            try:
                # -- DELETE CARRIER EQUATIONS --
                for region in regions:
                    ds.delete_equation(device=device_name, region=region, name="ElectronContinuity")
                    ds.delete_equation(device=device_name, region=region, name="HoleContinuity")
                for contact in contacts:
                    ds.delete_contact_equation(device=device_name, contact=contact, name="ElectronContinuity")
                    ds.delete_contact_equation(device=device_name, contact=contact, name="HoleContinuity")
                for interface in interfaces:
                    ds.delete_interface_equation(device=device_name, interface=interface, name="ElectronContinuity")
                    ds.delete_interface_equation(device=device_name, interface=interface, name="HoleContinuity")

                # -- SOLVE POISSON-ONLY SYSTEM --
                ds.solve(type="dc", absolute_error=1e12, relative_error=1e-8, maximum_iterations=10)
                poisson_error = ds.get_parameter(name="dc_error")
                print(f"    ✓ Potential solve step done. Error: {poisson_error:.4e}")

            finally:
                # -- EFFICIENTLY RESTORE CARRIER SYSTEM --
                restore_carrier_system(device_name)

            # ======================================================================
            # STEP 2: Solve for Carriers ONLY (Potential is frozen)
            # ======================================================================
            print("  [2/2] Solving for Carriers (ψ frozen)...")
            try:
                # -- DELETE POISSON EQUATION --
                for region in regions:
                    ds.delete_equation(device=device_name, region=region, name="PoissonEquation")
                for contact in contacts:
                    ds.delete_contact_equation(device=device_name, contact=contact, name="PoissonEquation")
                for interface in interfaces:
                    ds.delete_interface_equation(device=device_name, interface=interface, name="PoissonEquation")

                # -- SOLVE CARRIER-ONLY SYSTEM --
                ds.solve(type="dc", absolute_error=1e12, relative_error=1e-8, maximum_iterations=10)
                carrier_error = ds.get_parameter(name="dc_error")
                print(f"    ✓ Carrier solve step done. Error: {carrier_error:.4e}")

            finally:
                # -- EFFICIENTLY RESTORE POISSON SYSTEM --
                restore_poisson_system(device_name)

            # ======================================================================
            # STEP 3: Check for Global Convergence
            # ======================================================================
            if iteration >= min_check_iter:
                if poisson_error < convergence_tolerance and carrier_error < convergence_tolerance:
                    print("\n" + "=" * 70)
                    print(f"🎉 GUMMEL METHOD CONVERGED in {iteration + 1} iterations!")
                    print("=" * 70)
                    return True

        except ds.error as msg:
            # This will now catch the REAL error (e.g., "Convergence failure!").
            print(f"  ✗ ERROR during Gummel iteration {iteration + 1}: {msg}")
            diagnose_convergence_failure(device_name)
            return False

    # If the loop finishes without converging
    print("\n" + "=" * 70)
    print(f"✗ GUMMEL METHOD FAILED to converge within {max_gummel_iterations} iterations.")
    print("=" * 70)
    return False



def diagnose_convergence_failure(device_name):
    """
    Print detailed diagnostics when convergence fails.
    """
    print("\n" + "=" * 70)
    print("CONVERGENCE FAILURE DIAGNOSTICS")
    print("=" * 70)
    for region in ds.get_region_list(device=device_name):
        print(f"\nRegion: {region}")
        n_vals = ds.get_node_model_values(device=device_name, region=region, name="Electrons")
        p_vals = ds.get_node_model_values(device=device_name, region=region, name="Holes")
        V_vals = ds.get_node_model_values(device=device_name, region=region, name="Potential")
        print(f"  Electrons: min={min(n_vals):.2e}, max={max(n_vals):.2e}")
        print(f"  Holes:     min={min(p_vals):.2e}, max={max(p_vals):.2e}")
        print(f"  Potential: min={min(V_vals):.3f}V, max={max(V_vals):.3f}V")
        if min(n_vals) <= 0: print(f"  ✗ PROBLEM: Non-positive electrons!")
        if min(p_vals) <= 0: print(f"  ✗ PROBLEM: Non-positive holes!")


def verify_and_plot_equilibrium(device_name, width):
    """
    Performs checks and creates plots to verify the equilibrium solution.
    """
    # ... (This function remains unchanged, omitted for brevity)
    print("\n" + "=" * 70)
    print("STEP: VERIFYING AND PLOTTING EQUILIBRIUM SOLUTION")
    print("=" * 70)
    # ... (rest of the function)


def main():
    """Main execution with split physics setup."""
    default_mesh = "gmsh_diode2d.msh"
    width = 0.01
    mesh_file = sys.argv[1] if len(sys.argv) == 2 else default_mesh

    try:
        print("\nPHASE 1: DEVICE STRUCTURE SETUP")
        device_name = setup_photodiode_device(mesh_file)

        print("\nPHASE 2: POISSON PHYSICS SETUP")
        add_poisson_physics(device_name)


        print("\nPHASE 3: DRIFT-DIFFUSION PHYSICS SETUP")
        add_drift_diffusion_physics(device_name)

        # <<<<<<<<<<<<<<<< MODIFIED SECTION >>>>>>>>>>>>>>>>>>
        # PHASE 5: Solve equilibrium with gummel method
        print("\n" + "=" * 70)
        print("PHASE 5: EQUILIBRIUM SOLVE (GUMMEL METHOD)")
        print("=" * 70)
        success = solve_equilibrium_gummel(device_name)  # Use the new Gummel solver
        if not success:
            print("\n❌ FATAL: Equilibrium solve failed!")
            sys.exit(1)
        # <<<<<<<<<<<<<<<< END MODIFIED SECTION >>>>>>>>>>>>>>>>>>

        print("\nPHASE 6: VERIFICATION & VISUALIZATION")
        verify_and_plot_equilibrium(device_name, width)

        print("\n🎉 SIMULATION COMPLETE!")

    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()