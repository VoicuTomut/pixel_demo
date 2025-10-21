"""
2D Photodiode DEVSIM Simulation Setup - Enhanced & Corrected Version

This script imports a Gmsh mesh and prepares the complete DEVSIM device structure
with proper geometry setup, material parameters, and verification, ready for
drift-diffusion equation implementation.


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

        # CRITICAL: Re-set initial conditions
        print(f"    Re-initializing carrier concentrations...")
        set_initial_conditions(device_name, region)

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
        equation=f"Potential - {contact_name}_bias - {V_bi}"
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


def solve_poisson_only(device_name):
    """
    Solve ONLY Poisson's equation to get initial potential distribution.
    This is now possible since we split the physics setup.
    """
    print("\n" + "=" * 70)
    print("STEP: SOLVING POISSON EQUATION ONLY (INITIAL POTENTIAL)")
    print("=" * 70)

    try:
        # Solve with relaxed tolerance - it's just an initial guess
        print("\nSolving for electrostatic potential...")
        ds.solve(
            type="dc",
            absolute_error=1e12,
            relative_error=1e-8,
            maximum_iterations=50
        )
        print("✓ Successfully solved for initial potential distribution.")
        return True

    except ds.error as msg:
        print(f"✗ Poisson solve failed: {msg}")
        print("  This is not critical - continuing with charge neutrality guess...")
        return False



def setup_photodiode_device(mesh_file):
    """
    Complete setup of photodiode device structure in DEVSIM.

    This function:
    1. Imports the Gmsh mesh
    2. Creates contacts and interfaces
    3. Sets all material parameters and doping profiles
    4. Verifies the complete device structure

    Args:
        mesh_file: Path to Gmsh .msh file (MSH 2.2 ASCII format)

    Returns:
        device_name: String identifier for the device ('photodiode')

    Raises:
        FileNotFoundError: If mesh file doesn't exist
        ValueError: If expected regions/contacts/interfaces are missing
    """

    device_name = "photodiode"

    # =================================================================
    # PHYSICAL AND MATERIAL CONSTANTS
    # =================================================================

    print("\n" + "=" * 70)
    print("DEVSIM 2D PHOTODIODE DEVICE SETUP")
    print("=" * 70)

    # Physical constants (SI units converted to DEVSIM units)
    q = 1.602e-19  # Elementary charge [C]
    k_B = 1.381e-23  # Boltzmann constant [J/K]
    eps_0 = 8.854e-14  # Vacuum permittivity [F/cm]
    T = 300.0  # Temperature [K]
    V_t = k_B * T / q  # Thermal voltage [V] ≈ 0.0259 V

    # Silicon material properties at 300K
    eps_si = 11.7  # Relative permittivity (dimensionless)
    n_i = 1.0e10  # Intrinsic carrier concentration [cm^-3]

    # Mobility values [cm^2/(V·s)]
    mu_n = 1400.0  # Electron mobility (bulk silicon)
    mu_p = 450.0  # Hole mobility (bulk silicon)

    # SRH recombination lifetimes [s]
    tau_n = 1.0e-6  # Electron lifetime
    tau_p = 1.0e-6  # Hole lifetime

    # Doping concentrations [cm^-3]
    N_D_nplus = 1.0e19  # n+ cathode region (heavily doped)
    N_A_p = 1.0e16  # p-region (lightly doped)
    N_A_pplus = 1.0e19  # p+ anode substrate (heavily doped)

    print("\nPhysical Constants:")
    print(f"  Temperature (T):          {T} K")
    print(f"  Thermal voltage (V_t):    {V_t:.4f} V")
    print(f"  Intrinsic carrier (n_i):  {n_i:.2e} cm^-3")

    # =================================================================
    # 1. IMPORT GMSH MESH
    # =================================================================
    print("\n" + "=" * 70)
    print("STEP 1: IMPORTING GMSH MESH")
    print("=" * 70)

    # Verify file exists
    if not os.path.exists(mesh_file):
        raise FileNotFoundError(
            f"Mesh file not found: {mesh_file}\n"
            f"Please run gmesh_diode2d.py first to generate the mesh."
        )

    print(f"Reading mesh file: {mesh_file}")
    file_size = os.path.getsize(mesh_file) / 1024  # KB
    print(f"File size: {file_size:.1f} KB")

    # Import the mesh into DEVSIM
    # Note: Correct order is: 1) load mesh, 2) add regions, 3) add contacts, 4) add interfaces, 5) finalize, 6) create device
    try:
        try:
            existing_devices = ds.get_device_list()
            if device_name in existing_devices:
                ds.delete_device(device=device_name)
                print(f"  Deleted existing device '{device_name}' from previous run")
        except:
            pass
        try:
            ds.delete_mesh(mesh="diode_mesh")
            print("  Deleted existing mesh from previous run")
        except:
            pass  # Mesh doesn't exist, which is fine

        # Step 1: Load the mesh file
        ds.create_gmsh_mesh(mesh="diode_mesh", file=mesh_file)
        print("✓ Mesh file loaded successfully")

        # Step 2: Add regions to the mesh from physical groups (BEFORE finalizing)
        ds.add_gmsh_region(mesh="diode_mesh", gmsh_name="n_plus_region",
                           region="n_plus_region", material="Si")
        ds.add_gmsh_region(mesh="diode_mesh", gmsh_name="p_region",
                           region="p_region", material="Si")
        ds.add_gmsh_region(mesh="diode_mesh", gmsh_name="p_plus_region",
                           region="p_plus_region", material="Si")
        print("✓ Regions added to mesh")

        # Step 3: Add contacts to the mesh from physical groups (BEFORE finalizing)
        ds.add_gmsh_contact(mesh="diode_mesh", gmsh_name="cathode",
                            region="n_plus_region", name="cathode", material="metal")
        ds.add_gmsh_contact(mesh="diode_mesh", gmsh_name="anode",
                            region="p_plus_region", name="anode", material="metal")
        print("✓ Contacts added to mesh")

        # Step 4: Add interfaces between regions from physical groups (BEFORE finalizing)
        ds.add_gmsh_interface(mesh="diode_mesh", gmsh_name="pn_interface",
                              region0="n_plus_region", region1="p_region", name="pn_interface")
        ds.add_gmsh_interface(mesh="diode_mesh", gmsh_name="p_pplus_interface",
                              region0="p_region", region1="p_plus_region", name="p_pplus_interface")
        print("✓ Interfaces added to mesh")

        # Step 5: Finalize the mesh (after adding regions, contacts, and interfaces)
        ds.finalize_mesh(mesh="diode_mesh")
        print("✓ Mesh finalized")

        # Step 6: Create the device from the finalized mesh
        ds.create_device(mesh="diode_mesh", device=device_name)
        print(f"✓ Device '{device_name}' created from mesh")

    except Exception as e:
        raise RuntimeError(f"Failed to import mesh and create device: {e}")

    # Verify device was created
    device_list = ds.get_device_list()
    print(f"Devices in DEVSIM: {device_list}")

    if device_name not in device_list:
        if len(device_list) > 0:
            # Use the first available device
            actual_device = device_list[0]
            print(f"Warning: Expected device '{device_name}' not found.")
            print(f"Using device '{actual_device}' instead.")
            device_name = actual_device
        else:
            raise RuntimeError("No devices found in DEVSIM after mesh import!")

    # Verify all expected regions exist
    regions = ds.get_region_list(device=device_name)
    print(f"\nRegions found: {regions}")

    expected_regions = ['n_plus_region', 'p_region', 'p_plus_region']
    for region in expected_regions:
        if region not in regions:
            raise ValueError(
                f"CRITICAL: Expected region '{region}' not found in mesh!\n"
                f"Available regions: {regions}\n"
                f"Check that gmesh_diode2d.py generated the mesh correctly."
            )

    print("✓ All expected regions found")

    # Print mesh statistics for each region
    print("\nMesh Statistics:")
    total_nodes = 0
    for region in regions:
        node_indices = ds.get_node_model_values(
            device=device_name,
            region=region,
            name="node_index"
        )
        num_nodes = len(node_indices)
        total_nodes += num_nodes
        print(f"  • {region:20s}: {num_nodes:6d} nodes")
    print(f"  {'TOTAL':20s}: {total_nodes:6d} nodes")

    # =================================================================
    # 2. VERIFY CONTACTS
    # =================================================================
    print("\n" + "=" * 70)
    print("STEP 2: VERIFYING CONTACTS")
    print("=" * 70)

    # Contacts were already added during mesh import
    # Now we just verify they exist
    contacts = ds.get_contact_list(device=device_name)
    print(f"\nContacts found: {contacts}")

    expected_contacts = ['cathode', 'anode']
    for contact in expected_contacts:
        if contact not in contacts:
            raise ValueError(f"CRITICAL: Expected contact '{contact}' not found!")

    print("✓ All expected contacts verified")

    # =================================================================
    # 3. VERIFY INTERFACES
    # =================================================================
    print("\n" + "=" * 70)
    print("STEP 3: VERIFYING INTERFACES")
    print("=" * 70)

    # Interfaces were already added during mesh import
    # Now we just verify they exist
    interfaces = ds.get_interface_list(device=device_name)
    print(f"\nInterfaces found: {interfaces}")

    expected_interfaces = ['pn_interface', 'p_pplus_interface']
    for interface in expected_interfaces:
        if interface not in interfaces:
            raise ValueError(f"CRITICAL: Expected interface '{interface}' not found!")

    print("✓ All expected interfaces verified")

    # =================================================================
    # 4. SET GLOBAL DEVICE PARAMETERS
    # =================================================================
    print("\n" + "=" * 70)
    print("STEP 4: SETTING GLOBAL PARAMETERS")
    print("=" * 70)

    print("\nSetting device-wide physical constants...")
    global_params = {
        'T': T,
        'q': q,
        'k_B': k_B,
        'eps_0': eps_0,
        'V_t': V_t,
        'n_i': n_i
    }

    for param_name, param_value in global_params.items():
        ds.set_parameter(device=device_name, name=param_name, value=param_value)
        print(f"  • {param_name:10s} = {param_value:.4e}")

    print("✓ Global parameters set")

    # =================================================================
    # 5. SET REGION-SPECIFIC PARAMETERS
    # =================================================================
    print("\n" + "=" * 70)
    print("STEP 5: SETTING MATERIAL PARAMETERS AND DOPING")
    print("=" * 70)

    # Define region-specific doping profiles
    region_params = {
        'n_plus_region': {
            'N_D': N_D_nplus,
            'N_A': 0.0,
            'description': 'n+ cathode (heavily doped n-type)'
        },
        'p_region': {
            'N_D': 0.0,
            'N_A': N_A_p,
            'description': 'p-type active region (lightly doped)'
        },
        'p_plus_region': {
            'N_D': 0.0,
            'N_A': N_A_pplus,
            'description': 'p+ anode substrate (heavily doped p-type)'
        }
    }

    # Set parameters for each region
    for region in regions:
        print(f"\nRegion: {region}")
        print(f"  Description: {region_params[region]['description']}")

        # Silicon material properties (same for all regions)
        ds.set_parameter(device=device_name, region=region, name="eps_r", value=eps_si)
        ds.set_parameter(device=device_name, region=region, name="mu_n", value=mu_n)
        ds.set_parameter(device=device_name, region=region, name="mu_p", value=mu_p)
        ds.set_parameter(device=device_name, region=region, name="tau_n", value=tau_n)
        ds.set_parameter(device=device_name, region=region, name="tau_p", value=tau_p)

        # Doping concentrations (region-specific)
        doping = region_params[region]
        ds.set_parameter(device=device_name, region=region, name="N_D", value=doping['N_D'])
        ds.set_parameter(device=device_name, region=region, name="N_A", value=doping['N_A'])

        print(f"  Doping:")
        print(f"    - N_D (donor):    {doping['N_D']:.2e} cm^-3")
        print(f"    - N_A (acceptor): {doping['N_A']:.2e} cm^-3")
        print(f"  Transport:")
        print(f"    - μ_n: {mu_n:6.0f} cm²/(V·s)")
        print(f"    - μ_p: {mu_p:6.0f} cm²/(V·s)")
        print(f"  Recombination:")
        print(f"    - τ_n: {tau_n:.2e} s")
        print(f"    - τ_p: {tau_p:.2e} s")

    print("\n✓ Material parameters set for all regions")

    # =================================================================
    # 6. DEVICE STRUCTURE VERIFICATION
    # =================================================================
    print("\n" + "=" * 70)
    print("DEVICE STRUCTURE VERIFICATION")
    print("=" * 70)

    print(f"\n✓ Device name: '{device_name}'")
    print(f"✓ Number of regions:    {len(regions)}")
    print(f"✓ Number of contacts:   {len(contacts)}")
    print(f"✓ Number of interfaces: {len(interfaces)}")
    print(f"✓ Total mesh nodes:     {total_nodes}")

    # Verify we can retrieve parameters
    print("\nParameter Verification:")
    try:
        T_check = ds.get_parameter(device=device_name, name="T")
        n_i_check = ds.get_parameter(device=device_name, name="n_i")
        print(f"  • T  = {T_check} K ✓")
        print(f"  • n_i = {n_i_check:.2e} cm^-3 ✓")
    except Exception as e:
        print(f"  ✗ Warning: Could not verify parameters: {e}")

    # =================================================================
    # SETUP COMPLETE
    # =================================================================
    print("\n" + "=" * 70)
    print("📋 NEXT STEPS FOR SIMULATION")
    print("=" * 70)

    print("""
    ╔════════════════════════════════════════════════════════════════════╗
    ║ 💡 COMPLETE 2D SIMULATION WORKFLOW & PHYSICS GUIDE                  ║
    ╚════════════════════════════════════════════════════════════════════╝
      The following steps outline a robust strategy to solve the
      Drift-Diffusion equations for your 2D photodiode. All physics
      and boundary conditions are tailored to your gmsh_diode2d.py setup.

    ╔════════════════════════════════════════════════════════════════════╗
    ║ STEP 1: DEFINE VARIABLES, PHYSICS & EQUATIONS                      ║
    ╚════════════════════════════════════════════════════════════════════╝
      First, define the core components of the simulation for all regions.

      A. Primary Solution Variables (Node Models):
         - Potential: $\psi(x, y)$
         - Electrons: $n(x, y)$
         - Holes: $p(x, y)$

      B. Physics Models (Auxiliary Models):
         - Electric Field Vector: $\mathbf{E} = -\nabla\psi = -\left( \frac{\partial \psi}{\partial x}\mathbf{\hat{x}} + \frac{\partial \psi}{\partial y}\mathbf{\hat{y}} \right)$
         - Current Density Vectors (Drift-Diffusion):
           $$ \mathbf{J}_n = q\mu_n n \mathbf{E} + q D_n \nabla n $$
           $$ \mathbf{J}_p = q\mu_p p \mathbf{E} - q D_p \nabla p $$
         - Recombination Rate Models ($U = U_{SRH} + U_{Auger} + U_{rad}$):
           • SRH: $U_{SRH} = \\frac{np - n_i^2}{\\tau_p(n + n_1) + \\tau_n(p + p_1)}$
           • Auger: $U_{Auger} = (C_n n + C_p p)(np - n_i^2)$
           • Radiative: $U_{rad} = B(np - n_i^2)$

      C. Governing Equations (2D Steady-State):
         - Poisson's Equation: $\nabla\cdot(\epsilon\nabla\psi) = -q(p - n + N_D^+ - N_A^-)$
         - Electron Continuity: $\frac{1}{q}\nabla\cdot \mathbf{J}_n = U - G$
         - Hole Continuity: $-\frac{1}{q}\nabla\cdot \mathbf{J}_p = U - G$

      D. Boundary & Interface Conditions:
         - Ohmic Contacts (Dirichlet BCs):
           • On `cathode`: $\psi = V_{applied}$, $n = N_D$, $p = n_i^2/N_D$
           • On `anode`: $\psi = 0$, $p = N_A$, $n = n_i^2/N_A$
         - Insulating Surfaces (Neumann BCs):
           • On `top_surface`, `left_side`, `right_side`:
           • No normal E-field: $\nabla \psi \cdot \mathbf{\hat{n}} = 0$
           • Surface Recombination: $\mathbf{J}_n \cdot \mathbf{\hat{n}} = q S_n (n - n_{eq})$

    ╔════════════════════════════════════════════════════════════════════╗
    ║ STEP 2: SOLVE FOR POTENTIAL ONLY (INITIAL GUESS)                   ║
    ╚════════════════════════════════════════════════════════════════════╝
      Solve only Poisson's equation first to establish the initial
      built-in potential across the junctions. This provides a stable
      starting point for the fully coupled solver.

    ╔════════════════════════════════════════════════════════════════════╗
    ║ STEP 3: SOLVE FULLY COUPLED SYSTEM IN DARK (EQUILIBRIUM)           ║
    ╚════════════════════════════════════════════════════════════════════╝
      Solve the complete system (Poisson, Electron Continuity, Hole
      Continuity) with no applied voltage ($V_{applied} = 0$) and no
      optical generation ($G = 0$) to find the thermal equilibrium state.

    ╔════════════════════════════════════════════════════════════════════╗
    ║ STEP 4: SIMULATE DARK I-V CHARACTERISTICS                          ║
    ╚════════════════════════════════════════════════════════════════════╝
      Sweep the applied voltage on the `cathode` to get the dark I-V curve.

      1. Loop through voltage steps (e.g., from -1V to 0.8V).
      2. At each step, solve the fully coupled system with $G=0$.
      3. Calculate Terminal Current by integrating the normal component of
         the total current density vector along a contact boundary:
         $$ I_{dark}(V) = \int_{\text{anode}} (\mathbf{J}_n + \mathbf{J}_p) \cdot \mathbf{\hat{n}} \, dl $$
         (Note: The result is in Amperes per meter [A/m] for a 2D simulation).

    ╔════════════════════════════════════════════════════════════════════╗
    ║ STEP 5: SIMULATE ILLUMINATED CHARACTERISTICS                       ║
    ╚════════════════════════════════════════════════════════════════════╝
      Introduce an optical generation term ($G > 0$) to simulate the
      photodiode's response to light.

      1. Define an optical generation model based on depth ($y$) from the
         illuminated surface ($y=0$):
         $$ G(y, \lambda) = \alpha(\lambda)[1 - R_s(\lambda)]\Phi_0(\lambda)e^{-\alpha(\lambda)y} $$
      2. Repeat the voltage sweep from Step 4 with the non-zero $G(y, \lambda)$.
      3. Extract the total illuminated current $I_{light}(V)$.
      4. Calculate Photocurrent: $I_{ph}(V) = I_{light}(V) - I_{dark}(V)$.
      5. Analyze key metrics: Short-circuit current ($I_{sc}$), Open-circuit
         voltage ($V_{oc}$), External Quantum Efficiency (EQE), and Responsivity ($\mathcal{R}$).
    
    ╔════════════════════════════════════════════════════════════════════╗
    ║ 📚 REFERENCE RESOURCES                                             ║
    ╚════════════════════════════════════════════════════════════════════╝
      • DEVSIM Documentation: https://devsim.org
      • Example scripts: devsim/examples/diode/
      • Physics reference: S.M. Sze, "Physics of Semiconductor Devices"
    """)
    print("=" * 70)

    return device_name

def create_solution_variables(device_name, region):
    """
    Create the primary solution variables (unknowns) for the drift-diffusion equations.

    These are the three coupled variables we solve for:
    - Potential (ψ): Electrostatic potential [V]
    - Electrons (n): Electron concentration [cm^-3]
    - Holes (p): Hole concentration [cm^-3]

    Args:
        device_name: Device identifier
        region: Region name where variables are defined
    """
    print(f"  Creating solution variables in {region}...")

    # Create node solution variables (these are the unknowns)
    ds.node_solution(device=device_name, region=region, name="Potential")
    ds.node_solution(device=device_name, region=region, name="Electrons")
    ds.node_solution(device=device_name, region=region, name="Holes")

    print(f"    ✓ Potential, Electrons, Holes created")

def set_initial_conditions(device_name, region):
    """
    Set initial guesses for solution variables based on doping profile.

    Uses the charge neutrality approximation:
    - In n-type regions: n ≈ N_D, p ≈ n_i²/N_D
    - In p-type regions: p ≈ N_A, n ≈ n_i²/N_A
    - Potential initialized to zero (will be solved for)

    Args:
        device_name: Device identifier
        region: Region name
    """
    print(f"  Setting initial conditions in {region}...")

    # Get the number of nodes in the current region
    num_nodes = len(ds.get_node_model_values(device=device_name, region=region, name="node_index"))

    # Get doping and material parameters
    N_D = ds.get_parameter(device=device_name, region=region, name="N_D")
    N_A = ds.get_parameter(device=device_name, region=region, name="N_A")
    n_i = ds.get_parameter(device=device_name, name="n_i")

    # Net doping concentration
    N_net = N_D - N_A

    # Initial guess based on charge neutrality
    if N_net > 0:  # n-type region
        n_init = N_net
        p_init = n_i ** 2 / N_net
    else:  # p-type region
        p_init = abs(N_net)
        n_init = n_i ** 2 / abs(N_net)

    # CORRECTED: Create lists with a value for each node
    potential_vals = [0.0] * num_nodes
    electron_vals = [n_init] * num_nodes
    hole_vals = [p_init] * num_nodes

    # Set initial values using the correctly sized lists
    ds.set_node_values(device=device_name, region=region, name="Potential",
                       values=potential_vals)
    ds.set_node_values(device=device_name, region=region, name="Electrons",
                       values=electron_vals)
    ds.set_node_values(device=device_name, region=region, name="Holes",
                       values=hole_vals)

    print(f"    ✓ n_init = {n_init:.2e}, p_init = {p_init:.2e}")

def build_electric_field_model(device_name, region):
    """
    Build the electric field vector model: E = -∇ψ.

    The electric field component along each mesh edge is calculated as the
    negative potential difference divided by the edge length. This is a
    more general and physically correct approach for 2D/3D simulations.

    Args:
        device_name: Device identifier
        region: Region name
    """
    print(f"    Building electric field model...")

    # Makes Potential@n0 and Potential@n1 available on each edge
    ds.edge_from_node_model(
        device=device_name,
        region=region,
        node_model="Potential"
    )

    # CORRECTED: Use "ElectricField" for the general component along the edge
    # and "EdgeInverseLength" for 1/L to correctly calculate E = -dV/dL.
    ds.edge_model(
        device=device_name,
        region=region,
        name="ElectricField",
        equation="(Potential@n0 - Potential@n1) * EdgeInverseLength"
    )

    # Derivatives for the corrected electric field model
    ds.edge_model(
        device=device_name,
        region=region,
        name="ElectricField:Potential@n0",
        equation="EdgeInverseLength"
    )

    ds.edge_model(
        device=device_name,
        region=region,
        name="ElectricField:Potential@n1",
        equation="-EdgeInverseLength"
    )

    print(f"      ✓ Electric field E = -∇ψ created")


def build_drift_diffusion_model(device_name, region):
    """
    Builds Scharfetter-Gummel drift-diffusion with EXPLICIT DERIVATIVES.
    CRITICAL: DEVSIM cannot auto-differentiate Bernoulli functions!
    """
    print(f"    Building Scharfetter-Gummel drift-diffusion models...")

    # Get material parameters
    mu_n = ds.get_parameter(device=device_name, region=region, name="mu_n")
    mu_p = ds.get_parameter(device=device_name, region=region, name="mu_p")
    q = ds.get_parameter(device=device_name, name="q")
    V_t = ds.get_parameter(device=device_name, name="V_t")

    # Potential difference over thermal voltage
    ds.edge_model(
        device=device_name,
        region=region,
        name="V_diff_over_V_t",
        equation="(Potential@n0 - Potential@n1)/V_t"
    )

    # Make carrier concentrations available on edges
    ds.edge_from_node_model(device=device_name, region=region, node_model="Electrons")
    ds.edge_from_node_model(device=device_name, region=region, node_model="Holes")

    # ============================================================
    # ELECTRON CURRENT with DERIVATIVES
    # ============================================================
    ds.edge_model(
        device=device_name,
        region=region,
        name="ElectronCurrent",
        equation=f"{q}*{mu_n}*{V_t}*EdgeInverseLength*(Electrons@n0*B(V_diff_over_V_t) - Electrons@n1*B(-V_diff_over_V_t))"
    )

    # Derivative w.r.t. Potential@n0
    ds.edge_model(
        device=device_name,
        region=region,
        name="ElectronCurrent:Potential@n0",
        equation=f"{q}*{mu_n}*EdgeInverseLength*(Electrons@n0*dBdx(V_diff_over_V_t) + Electrons@n1*dBdx(-V_diff_over_V_t))"
    )

    # Derivative w.r.t. Potential@n1
    ds.edge_model(
        device=device_name,
        region=region,
        name="ElectronCurrent:Potential@n1",
        equation=f"-{q}*{mu_n}*EdgeInverseLength*(Electrons@n0*dBdx(V_diff_over_V_t) + Electrons@n1*dBdx(-V_diff_over_V_t))"
    )

    # Derivative w.r.t. Electrons@n0
    ds.edge_model(
        device=device_name,
        region=region,
        name="ElectronCurrent:Electrons@n0",
        equation=f"{q}*{mu_n}*{V_t}*EdgeInverseLength*B(V_diff_over_V_t)"
    )

    # Derivative w.r.t. Electrons@n1
    ds.edge_model(
        device=device_name,
        region=region,
        name="ElectronCurrent:Electrons@n1",
        equation=f"-{q}*{mu_n}*{V_t}*EdgeInverseLength*B(-V_diff_over_V_t)"
    )

    # ============================================================
    # HOLE CURRENT with DERIVATIVES
    # ============================================================
    ds.edge_model(
        device=device_name,
        region=region,
        name="HoleCurrent",
        equation=f"-{q}*{mu_p}*{V_t}*EdgeInverseLength*(Holes@n1*B(V_diff_over_V_t) - Holes@n0*B(-V_diff_over_V_t))"
    )

    # Derivative w.r.t. Potential@n0
    ds.edge_model(
        device=device_name,
        region=region,
        name="HoleCurrent:Potential@n0",
        equation=f"-{q}*{mu_p}*EdgeInverseLength*(Holes@n1*dBdx(V_diff_over_V_t) + Holes@n0*dBdx(-V_diff_over_V_t))"
    )

    # Derivative w.r.t. Potential@n1
    ds.edge_model(
        device=device_name,
        region=region,
        name="HoleCurrent:Potential@n1",
        equation=f"{q}*{mu_p}*EdgeInverseLength*(Holes@n1*dBdx(V_diff_over_V_t) + Holes@n0*dBdx(-V_diff_over_V_t))"
    )

    # Derivative w.r.t. Holes@n0
    ds.edge_model(
        device=device_name,
        region=region,
        name="HoleCurrent:Holes@n0",
        equation=f"{q}*{mu_p}*{V_t}*EdgeInverseLength*B(-V_diff_over_V_t)"
    )

    # Derivative w.r.t. Holes@n1
    ds.edge_model(
        device=device_name,
        region=region,
        name="HoleCurrent:Holes@n1",
        equation=f"-{q}*{mu_p}*{V_t}*EdgeInverseLength*B(V_diff_over_V_t)"
    )

    print(f"      ✓ Scharfetter-Gummel currents J_n and J_p created WITH DERIVATIVES")


def build_recombination_model(device_name, region, srh=True, radiative=True, auger=True):
    """
    Builds recombination models WITH REQUIRED DERIVATIVES.
    """
    print(f"    Building recombination models...")

    # Get parameters
    n_i = ds.get_parameter(device=device_name, name="n_i")
    tau_n = ds.get_parameter(device=device_name, region=region, name="tau_n")
    tau_p = ds.get_parameter(device=device_name, region=region, name="tau_p")
    n_i_sq = n_i ** 2

    # 1. SRH Recombination
    if srh:
        n1, p1 = n_i, n_i

        ds.node_model(
            device=device_name, region=region, name="U_SRH",
            equation=f"(Electrons*Holes - {n_i_sq}) / ({tau_p}*(Electrons + {n1}) + {tau_n}*(Holes + {p1}))"
        )

        # SRH derivatives
        ds.node_model(
            device=device_name, region=region, name="U_SRH:Electrons",
            equation=f"(Holes*({tau_p}*(Electrons+{n1})+{tau_n}*(Holes+{p1})) - (Electrons*Holes-{n_i_sq})*{tau_p}) / (({tau_p}*(Electrons+{n1})+{tau_n}*(Holes+{p1}))^2)"
        )

        ds.node_model(
            device=device_name, region=region, name="U_SRH:Holes",
            equation=f"(Electrons*({tau_p}*(Electrons+{n1})+{tau_n}*(Holes+{p1})) - (Electrons*Holes-{n_i_sq})*{tau_n}) / (({tau_p}*(Electrons+{n1})+{tau_n}*(Holes+{p1}))^2)"
        )

        ds.node_model(
            device=device_name, region=region, name="U_SRH:Potential",
            equation="0"
        )

        print(f"      ✓ SRH recombination added WITH DERIVATIVES")

    # 2. Radiative Recombination
    if radiative:
        B_rad = 1.0e-14

        ds.node_model(
            device=device_name, region=region, name="U_radiative",
            equation=f"{B_rad}*(Electrons*Holes - {n_i_sq})"
        )

        ds.node_model(
            device=device_name, region=region, name="U_radiative:Electrons",
            equation=f"{B_rad}*Holes"
        )

        ds.node_model(
            device=device_name, region=region, name="U_radiative:Holes",
            equation=f"{B_rad}*Electrons"
        )

        ds.node_model(
            device=device_name, region=region, name="U_radiative:Potential",
            equation="0"
        )

        print(f"      ✓ Radiative recombination added WITH DERIVATIVES")

    # 3. Auger Recombination
    if auger:
        C_n, C_p = 2.8e-31, 9.9e-32

        ds.node_model(
            device=device_name, region=region, name="U_Auger",
            equation=f"({C_n}*Electrons + {C_p}*Holes)*(Electrons*Holes - {n_i_sq})"
        )

        ds.node_model(
            device=device_name, region=region, name="U_Auger:Electrons",
            equation=f"{C_n}*(Electrons*Holes - {n_i_sq}) + ({C_n}*Electrons + {C_p}*Holes)*Holes"
        )

        ds.node_model(
            device=device_name, region=region, name="U_Auger:Holes",
            equation=f"{C_p}*(Electrons*Holes - {n_i_sq}) + ({C_n}*Electrons + {C_p}*Holes)*Electrons"
        )

        ds.node_model(
            device=device_name, region=region, name="U_Auger:Potential",
            equation="0"
        )

        print(f"      ✓ Auger recombination added WITH DERIVATIVES")

    # Total Recombination
    U_total = "0"
    dU_n = "0"
    dU_p = "0"

    if srh:
        U_total += " + U_SRH"
        dU_n += " + U_SRH:Electrons"
        dU_p += " + U_SRH:Holes"
    if radiative:
        U_total += " + U_radiative"
        dU_n += " + U_radiative:Electrons"
        dU_p += " + U_radiative:Holes"
    if auger:
        U_total += " + U_Auger"
        dU_n += " + U_Auger:Electrons"
        dU_p += " + U_Auger:Holes"

    ds.node_model(
        device=device_name, region=region,
        name="Recombination",
        equation=U_total
    )

    ds.node_model(
        device=device_name, region=region,
        name="Recombination:Electrons",
        equation=dU_n
    )

    ds.node_model(
        device=device_name, region=region,
        name="Recombination:Holes",
        equation=dU_p
    )

    ds.node_model(
        device=device_name, region=region,
        name="Recombination:Potential",
        equation="0"
    )

def construct_poisson_eq(device_name, region):
    """
    Construct Poisson's equation:
    ∇·(ε∇ψ) = -q(p - n + N_D - N_A)

    Args:
        device_name: Device identifier
        region: Region name
    """
    print(f"    Constructing Poisson's equation...")

    # Get parameters
    q = ds.get_parameter(device=device_name, name="q")
    eps_0 = ds.get_parameter(device=device_name, name="eps_0")
    eps_r = ds.get_parameter(device=device_name, region=region, name="eps_r")
    N_D = ds.get_parameter(device=device_name, region=region, name="N_D")
    N_A = ds.get_parameter(device=device_name, region=region, name="N_A")

    epsilon = eps_0 * eps_r

    # Net doping (ionized charge)
    ds.node_model(
        device=device_name,
        region=region,
        name="NetDoping",
        equation=f"{N_D} - {N_A}"
    )

    # Space charge density: ρ = q(p - n + N_D - N_A)
    ds.node_model(
        device=device_name,
        region=region,
        name="SpaceCharge",
        equation=f"{q}*(Holes - Electrons + NetDoping)"
    )

    # Derivatives
    ds.node_model(
        device=device_name,
        region=region,
        name="SpaceCharge:Electrons",
        equation=f"-{q}"
    )

    ds.node_model(
        device=device_name,
        region=region,
        name="SpaceCharge:Holes",
        equation=f"{q}"
    )

    # Electric flux: D = ε∇ψ (edge model)
    ds.edge_model(
        device=device_name,
        region=region,
        name="ElectricFlux",
        equation=f"{epsilon}*ElectricField"
    )

    ds.edge_model(
        device=device_name,
        region=region,
        name="ElectricFlux:Potential@n0",
        equation=f"{epsilon}*ElectricField:Potential@n0"
    )

    ds.edge_model(
        device=device_name,
        region=region,
        name="ElectricFlux:Potential@n1",
        equation=f"{epsilon}*ElectricField:Potential@n1"
    )

    # Add equation to solver
    ds.equation(
        device=device_name,
        region=region,
        name="PoissonEquation",
        variable_name="Potential",
        edge_model="ElectricFlux",
        edge_volume_model="",
        node_model="SpaceCharge",
        variable_update="default"
    )

    print(f"      ✓ Poisson equation: ∇·(ε∇ψ) = -q(p - n + N_D - N_A)")

def construct_electron_continuity_eq(device_name, region):
    """
    Construct electron continuity equation: ∇·J_n - q(U - G) = 0
    """
    print(f"    Constructing electron continuity equation...")

    # Generation rate (initially zero, can be set later for illumination)
    ds.node_model(
        device=device_name,
        region=region,
        name="OpticalGeneration",
        equation="0.0"
    )

    # Net recombination term for the equation: S = -q * (U - G)
    ds.node_model(
        device=device_name,
        region=region,
        name="ElectronGenerationSource",
        equation="-q * (Recombination - OpticalGeneration)"
    )

    # Derivatives of the source term
    ds.node_model(device=device_name, region=region, name="ElectronGenerationSource:Electrons",
                  equation="-q * Recombination:Electrons")
    ds.node_model(device=device_name, region=region, name="ElectronGenerationSource:Holes",
                  equation="-q * Recombination:Holes")
    ds.node_model(device=device_name, region=region, name="ElectronGenerationSource:Potential",
                  equation="-q * Recombination:Potential")


    # Add equation to solver
    ds.equation(
        device=device_name,
        region=region,
        name="ElectronContinuity",
        variable_name="Electrons",
        edge_model="ElectronCurrent",
        node_model="ElectronGenerationSource",
        variable_update="positive"
    )

    print(f"      ✓ Electron continuity: ∇·J_n - q(U-G) = 0")

def construct_hole_continuity_eq(device_name, region):
    """
    Construct hole continuity equation: ∇·J_p + q(U - G) = 0
    """
    print(f"    Constructing hole continuity equation...")

    # Net recombination term for the equation: S = q * (U - G)
    # OpticalGeneration is already defined in the electron continuity function
    ds.node_model(
        device=device_name,
        region=region,
        name="HoleGenerationSource",
        equation="q * (Recombination - OpticalGeneration)"
    )

    # Derivatives of the source term
    ds.node_model(device=device_name, region=region, name="HoleGenerationSource:Electrons",
                  equation="q * Recombination:Electrons")
    ds.node_model(device=device_name, region=region, name="HoleGenerationSource:Holes",
                  equation="q * Recombination:Holes")
    ds.node_model(device=device_name, region=region, name="HoleGenerationSource:Potential",
                  equation="q * Recombination:Potential")


    # Add equation to solver
    ds.equation(
        device=device_name,
        region=region,
        name="HoleContinuity",
        variable_name="Holes",
        edge_model="HoleCurrent",
        node_model="HoleGenerationSource",
        variable_update="positive"
    )

    print(f"      ✓ Hole continuity: ∇·J_p + q(U-G) = 0")

def add_ohmic_contact(device_name, contact_name, region_name, bias=0.0):
    """
    Add ohmic contact boundary conditions.

    For ohmic contacts:
    - ψ = V_applied + V_bi (built-in potential)
    - n = N_D (for n-type contact)
    - p = n_i²/N_D (for n-type contact)

    Args:
        device_name: Device identifier
        contact_name: Name of contact (e.g., 'cathode', 'anode')
        region_name: Associated region name
        bias: Applied bias voltage [V]
    """
    print(f"    Adding ohmic contact: {contact_name}")

    # Get parameters
    n_i = ds.get_parameter(device=device_name, name="n_i")
    N_D = ds.get_parameter(device=device_name, region=region_name, name="N_D")
    N_A = ds.get_parameter(device=device_name, region=region_name, name="N_A")
    V_t = ds.get_parameter(device=device_name, name="V_t")

    # Determine contact type and equilibrium carrier concentrations
    if N_D > N_A:  # n-type contact
        n_contact = N_D
        p_contact = n_i ** 2 / N_D
        V_bi = V_t * math.log(N_D / n_i)
    else:  # p-type contact
        p_contact = N_A
        n_contact = n_i ** 2 / N_A
        V_bi = -V_t * math.log(N_A / n_i)

    # Set contact bias
    ds.set_parameter(device=device_name, name=f"{contact_name}_bias", value=bias)

    # Contact potential (including built-in potential)
    ds.contact_node_model(
        device=device_name,
        contact=contact_name,
        name=f"{contact_name}_bc",
        equation=f"Potential - {contact_name}_bias - {V_bi}"
    )

    # Derivative of contact potential BC
    ds.contact_node_model(
        device=device_name,
        contact=contact_name,
        name=f"{contact_name}_bc:Potential",
        equation="1.0"
    )

    # Contact carrier concentrations
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

    # Add contact equations (NO variable_name parameter!)
    ds.contact_equation(
        device=device_name,
        contact=contact_name,
        name="PoissonEquation",
        node_model=f"{contact_name}_bc",
        edge_charge_model="ElectricFlux"
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

    print(f"      ✓ Ohmic contact {contact_name}: V={bias}V, n={n_contact:.2e}, p={p_contact:.2e}")


def add_interface_conditions(device_name, interface_name):
    """
    Add continuity conditions at semiconductor-semiconductor interfaces.

    At interfaces between regions (e.g., p-n junction):
    - ψ continuous
    - n, p continuous
    - ε∇ψ continuous (Gauss's law)
    - J_n, J_p continuous (current conservation)

    Args:
        device_name: Device identifier
        interface_name: Name of interface (e.g., 'pn_interface')
    """
    print(f"    Adding interface conditions: {interface_name}")

    # Create continuity models for potential
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

    # Create continuity models for electrons
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

    # Create continuity models for holes
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

    # Apply continuous potential condition
    ds.interface_equation(
        device=device_name,
        interface=interface_name,
        name="PoissonEquation",
        interface_model="continuous_potential",
        type="continuous"
    )

    # Apply continuous electron concentration condition
    ds.interface_equation(
        device=device_name,
        interface=interface_name,
        name="ElectronContinuity",
        interface_model="continuous_electrons",
        type="continuous"
    )

    # Apply continuous hole concentration condition
    ds.interface_equation(
        device=device_name,
        interface=interface_name,
        name="HoleContinuity",
        interface_model="continuous_holes",
        type="continuous"
    )

    print(f"      ✓ Interface conditions: continuous ψ, n, p")

def add_physics(device_name):
    """
    STEP 1: DEFINE VARIABLES, PHYSICS & EQUATIONS
    :param device_name:
    :return:
    """

    regions = ds.get_region_list(device=device_name)
    contacts = ds.get_contact_list(device=device_name)
    interfaces = ds.get_interface_list(device=device_name)

    print(f"\nConfiguring physics for {len(regions)} regions: {regions}")
    for region in regions:
        print(f"\n{'─' * 70}")
        print(f"Region: {region}")
        print(f"{'─' * 70}")

        # Step 1: Create solution variables
        create_solution_variables(device_name, region)
        # Step 2: Set initial conditions
        set_initial_conditions(device_name, region)

        #Physics Models
        build_electric_field_model(device_name,  region)
        build_drift_diffusion_model(device_name,  region)
        build_recombination_model(device_name,  region)

        #Construct equations
        construct_poisson_eq(device_name,region)
        construct_electron_continuity_eq(device_name,region)
        construct_hole_continuity_eq(device_name, region)

    print(f"\n{'─' * 70}")
    print("CONTACT BOUNDARY CONDITIONS")
    print(f"{'─' * 70}")

    for contact in contacts:
        # Get the region associated with this contact
        if contact == 'cathode':
            region = 'n_plus_region'
            bias = 0.0  # Initial bias
        elif contact == 'anode':
            region = 'p_plus_region'
            bias = 0.0  # Ground reference
        else:
            print(f"Warning: Unknown contact {contact}, skipping...")
            continue

        add_ohmic_contact(device_name, contact, region, bias)

    print(f"\n{'─' * 70}")
    print("INTERFACE CONDITIONS")
    print(f"{'─' * 70}")

    for interface in interfaces:
        add_interface_conditions(device_name, interface)

    print("\n" + "=" * 70)
    print("✓ PHYSICS SETUP COMPLETE")
    print("=" * 70)
    print("\nDevice is ready for solving!")
    print("Next steps:")
    print("  1. Solve equilibrium (V=0, G=0)")
    print("  2. Solve dark I-V characteristics")
    print("  3. Add optical generation and solve illuminated characteristics")



    return device_name




def solve_equilibrium(device_name):
    """
    Solves the fully coupled drift-diffusion system for thermal equilibrium.
    Uses robust multi-stage approach with forced carrier initialization.
    """
    print("\n" + "=" * 70)
    print("STEP: SOLVING FULLY COUPLED SYSTEM IN EQUILIBRIUM")
    print("=" * 70)

    # CRITICAL: Verify and fix carrier concentrations before solving
    print("\nVerifying initial carrier concentrations...")
    regions = ds.get_region_list(device=device_name)

    for region in regions:
        try:
            n_vals = ds.get_node_model_values(device=device_name, region=region, name="Electrons")
            p_vals = ds.get_node_model_values(device=device_name, region=region, name="Holes")
            n_min = min(n_vals)
            p_min = min(p_vals)
            n_max = max(n_vals)
            p_max = max(p_vals)
            num_nodes = len(n_vals)

            print(f"  {region}:")
            print(f"    Nodes:     {num_nodes}")
            print(f"    Electrons: min={n_min:.2e}, max={n_max:.2e}")
            print(f"    Holes:     min={p_min:.2e}, max={p_max:.2e}")

            # Check for problematic values
            if n_min <= 0 or p_min <= 0 or n_min < 1e-10 or p_min < 1e-10:
                print(f"    ⚠ WARNING: Non-positive or too-small carriers detected!")
                print(f"    Forcing re-initialization of {region}...")

                N_D = ds.get_parameter(device=device_name, region=region, name="N_D")
                N_A = ds.get_parameter(device=device_name, region=region, name="N_A")
                n_i = ds.get_parameter(device=device_name, name="n_i")

                N_net = N_D - N_A

                if N_net > 0:  # n-type
                    n_init = max(N_net, 1e10)
                    p_init = max(n_i ** 2 / N_net, 1e5)
                else:  # p-type
                    p_init = max(abs(N_net), 1e10)
                    n_init = max(n_i ** 2 / abs(N_net), 1e5)

                print(f"    Setting: n={n_init:.2e}, p={p_init:.2e} for ALL {num_nodes} nodes")

                # CRITICAL FIX: Create array with values for ALL nodes
                n_init_array = [n_init] * num_nodes
                p_init_array = [p_init] * num_nodes

                ds.set_node_values(
                    device=device_name,
                    region=region,
                    name="Electrons",

                    values=n_init_array
                )

                ds.set_node_values(
                    device=device_name,
                    region=region,
                    name="Holes",

                    values=p_init_array
                )

                # Verify the fix worked
                n_check = ds.get_node_model_values(device=device_name, region=region, name="Electrons")
                p_check = ds.get_node_model_values(device=device_name, region=region, name="Holes")
                print(f"    After fix: n_min={min(n_check):.2e}, p_min={min(p_check):.2e}")

                if min(n_check) <= 0 or min(p_check) <= 0:
                    print(f"    ✗ ERROR: Still have zero carriers after fix!")
                    return False
            else:
                print(f"    ✓ Carriers are positive and valid")

        except Exception as e:
            print(f"    ✗ Error checking/fixing {region}: {e}")
            import traceback
            traceback.print_exc()
            return False

    print("\n" + "─" * 70)
    print("Starting multi-stage solve...")
    print("─" * 70)

    try:
        print("\nStage 1: Very relaxed convergence...")
        ds.solve(type="dc", absolute_error=1e15, relative_error=1e-3, maximum_iterations=100)
        print("✓ Stage 1 complete")

        print("\nStage 2: Medium convergence...")
        ds.solve(type="dc", absolute_error=1e12, relative_error=1e-6, maximum_iterations=100)
        print("✓ Stage 2 complete")

        print("\nStage 3: Tighter convergence...")
        ds.solve(type="dc", absolute_error=1e10, relative_error=1e-10, maximum_iterations=100)
        print("✓ Stage 3 complete")

        print("\nStage 4: Final tight convergence...")
        ds.solve(type="dc", absolute_error=1e8, relative_error=1e-12, maximum_iterations=100)
        print("✓ Successfully solved for thermal equilibrium.")

    except ds.error as msg:
        print(f"\n✗ Convergence failed: {msg}")
        try:
            ds.write_devices(file="photodiode_FAILED.dat", type="tecplot")
            print("  Wrote failed state to photodiode_FAILED.dat")
        except:
            pass
        return False

    return True


def verify_and_plot_equilibrium(device_name, width):
    """
    Performs checks and creates plots to verify the equilibrium solution.

    Args:
        device_name: The device to analyze.
        width: The device width in cm (from gmesh_diode2d.py) for the line cut.
    """
    print("\n" + "="*70)
    print("STEP: VERIFYING AND PLOTTING EQUILIBRIUM SOLUTION")
    print("="*70)

    # 1. Verification: Check contact charge balance
    try:
        charge_anode = ds.get_contact_charge(device=device_name, contact="anode", equation="PoissonEquation")
        charge_cathode = ds.get_contact_charge(device=device_name, contact="cathode", equation="PoissonEquation")
        print("✓ Verification Checks:")
        print(f"  - Anode Charge:   {charge_anode:.4e} C/m")
        print(f"  - Cathode Charge: {charge_cathode:.4e} C/m")
        if abs(charge_anode) > 1e-30:  # Check if charge is non-zero
            if abs(charge_anode + charge_cathode) / abs(charge_anode) < 1e-5:
                print("  - INFO: Charge balance is excellent. 👍")
            else:
                print("  - WARNING: Significant charge imbalance detected. 🚩")
        else:
            print("  - WARNING: Contact charge is zero. The simulation may have failed to converge. 🚩")
    except ds.error as msg:
        print(f"✗ Could not retrieve contact charge: {msg}")


    # 2. Visualization: Save 2D data for ParaView
    print("\n✓ Visualization:")
    try:
        ds.write_devices(file="photodiode_equilibrium.vtk", type="vtk")
        print("  - Saved 2D solution to photodiode_equilibrium.vtk for ParaView.")
    except ds.error as msg:
        print(f"✗ Failed to write VTK file: {msg}")


    # 3. Plotting: Create 1D line cuts
    print("  - Generating 1D line cut plots...")
    try:
        # Get all nodes and their y-coordinates for all regions
        all_y_coords = []
        all_potential = []
        all_electrons = []
        all_holes = []

        for region in ds.get_region_list(device=device_name):
            # Get node coordinates and variables
            x_coords = ds.get_node_model_values(device=device_name, region=region, name="x")
            y_coords = ds.get_node_model_values(device=device_name, region=region, name="y")
            potential = ds.get_node_model_values(device=device_name, region=region, name="Potential")
            electrons = ds.get_node_model_values(device=device_name, region=region, name="Electrons")
            holes = ds.get_node_model_values(device=device_name, region=region, name="Holes")

            # Filter for nodes in the center of the device
            center_x = width / 2.0
            tolerance = 1e-9
            for i, x in enumerate(x_coords):
                if abs(x - center_x) < tolerance:
                    all_y_coords.append(y_coords[i])
                    all_potential.append(potential[i])
                    all_electrons.append(electrons[i])
                    all_holes.append(holes[i])

        if not all_y_coords:
            print("  - WARNING: Could not find any nodes at the device center for plotting. 🚩")
            return

        # Sort the data by y-coordinate for clean plotting
        sorted_indices = np.argsort(all_y_coords)
        y = np.array(all_y_coords)[sorted_indices]
        V = np.array(all_potential)[sorted_indices]
        n = np.array(all_electrons)[sorted_indices]
        p = np.array(all_holes)[sorted_indices]

        # Create plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle('Equilibrium Solution (Vertical Cut at x = width/2)', fontsize=16)

        # Potential Plot
        ax1.plot(y * 1e4, V, 'r-', lw=2) # Convert y from cm to µm
        ax1.set_xlabel('Depth (µm)')
        ax1.set_ylabel('Potential (V)')
        ax1.set_title('Electrostatic Potential')
        ax1.grid(True)

        # Carrier Concentration Plot
        ax2.semilogy(y * 1e4, n, 'b-', lw=2, label='Electrons ($n$)')
        ax2.semilogy(y * 1e4, p, 'g-', lw=2, label='Holes ($p$)')
        ax2.set_xlabel('Depth (µm)')
        ax2.set_ylabel('Carrier Concentration (cm⁻³)')
        ax2.set_title('Carrier Concentrations (Log Scale)')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig("photodiode_equilibrium_plots.png")
        print("  - Saved 1D plots to photodiode_equilibrium_plots.png.")
        # plt.show() # Uncomment to display plots interactively

    except Exception as e:
        print(f"✗ An error occurred during plotting: {e}")


def main():
    """Main execution with split physics setup."""
    import sys
    import os

    default_mesh = "gmsh_diode2d.msh"
    width = 0.01

    if len(sys.argv) > 2:
        print("=" * 70)
        print("ERROR: Too many arguments")
        print("=" * 70)
        print(f"\nUsage: python {sys.argv[0]} [mesh_file]")
        sys.exit(1)

    mesh_file = sys.argv[1] if len(sys.argv) == 2 else default_mesh

    try:
        # PHASE 1: Setup device structure
        print("\n" + "=" * 70)
        print("PHASE 1: DEVICE STRUCTURE SETUP")
        print("=" * 70)
        device_name = setup_photodiode_device(mesh_file)
        print(f"\n✓ Device '{device_name}' structure ready!")

        # PHASE 2: Add Poisson physics only
        print("\n" + "=" * 70)
        print("PHASE 2: POISSON PHYSICS SETUP")
        print("=" * 70)
        add_poisson_physics(device_name)

        # PHASE 3: Solve Poisson only (optional but recommended)
        print("\n" + "=" * 70)
        print("PHASE 3: INITIAL POTENTIAL SOLVE")
        print("=" * 70)
        solve_poisson_only(device_name)

        # PHASE 4: Add drift-diffusion physics
        print("\n" + "=" * 70)
        print("PHASE 4: DRIFT-DIFFUSION PHYSICS SETUP")
        print("=" * 70)
        add_drift_diffusion_physics(device_name)

        # PHASE 5: Solve equilibrium (fully coupled)
        print("\n" + "=" * 70)
        print("PHASE 5: EQUILIBRIUM SOLVE")
        print("=" * 70)
        success = solve_equilibrium(device_name)
        if not success:
            print("\n❌ FATAL: Equilibrium solve failed!")
            sys.exit(1)

        # PHASE 6: Verify and visualize
        print("\n" + "=" * 70)
        print("PHASE 6: VERIFICATION & VISUALIZATION")
        print("=" * 70)
        verify_and_plot_equilibrium(device_name, width)

        print("\n" + "=" * 70)
        print("🎉 SIMULATION COMPLETE!")
        print("=" * 70)
        print("\n📊 Output files:")
        print("  • photodiode_equilibrium.vtk")
        print("  • photodiode_equilibrium_plots.png")
        print("=" * 70 + "\n")

    except FileNotFoundError as e:
        print(f"\n❌ FILE ERROR: {e}")
        print(f"To generate mesh: python gmesh_diode2d.py\n")
        sys.exit(1)
    except ValueError as e:
        print(f"\n❌ STRUCTURE ERROR: {e}\n")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
