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

    # Set initial values
    ds.set_node_values(device=device_name, region=region, name="Potential",
                       init_from="Potential", values=[0.0])
    ds.set_node_values(device=device_name, region=region, name="Electrons",
                       init_from="Electrons", values=[n_init])
    ds.set_node_values(device=device_name, region=region, name="Holes",
                       init_from="Holes", values=[p_init])

    print(f"    ✓ n_init = {n_init:.2e}, p_init = {p_init:.2e}")

def build_electric_field_model(device_name, region):
    """
        Build electric field vector model: E = -∇ψ

        The electric field is calculated as the negative gradient of the potential:
        E_x = -∂ψ/∂x
        E_y = -∂ψ/∂y

        Args:
            device_name: Device identifier
            region: Region name
        """
    print(f"    Building electric field model...")

    # Electric field components (edge models for better accuracy)
    # E_x component
    ds.edge_from_node_model(
        device=device_name,
        region=region,
        node_model="Potential"
    )

    ds.edge_model(
        device=device_name,
        region=region,
        name="ElectricField_x",
        equation="-(Potential@n1 - Potential@n0) * EdgeCouple"
    )

    # Derivatives for node-based electric field (used in drift-diffusion)
    ds.edge_model(
        device=device_name,
        region=region,
        name="ElectricField_x:Potential@n0",
        equation="EdgeCouple"
    )

    ds.edge_model(
        device=device_name,
        region=region,
        name="ElectricField_x:Potential@n1",
        equation="-EdgeCouple"
    )

    print(f"      ✓ Electric field E = -∇ψ created")

def build_drift_diffusion_model(device_name, region):
    """
    Build drift-diffusion current density models:
    J_n = q·μ_n·n·E + q·D_n·∇n  (electron current)
    J_p = q·μ_p·p·E - q·D_p·∇p  (hole current)

    Args:
        device_name: Device identifier
        region: Region name
    """
    print(f"    Building drift-diffusion models...")

    # Get parameters
    mu_n = ds.get_parameter(device=device_name, region=region, name="mu_n")
    mu_p = ds.get_parameter(device=device_name, region=region, name="mu_p")
    T = ds.get_parameter(device=device_name, name="T")
    q = ds.get_parameter(device=device_name, name="q")
    k_B = ds.get_parameter(device=device_name, name="k_B")

    # Calculate diffusion coefficients using Einstein relation
    V_t = k_B * T / q  # Thermal voltage
    D_n = mu_n * V_t
    D_p = mu_p * V_t

    # Store diffusion coefficients as parameters
    ds.set_parameter(device=device_name, region=region, name="D_n", value=D_n)
    ds.set_parameter(device=device_name, region=region, name="D_p", value=D_p)

    # Electron current density (edge model)
    # Drift component: q·μ_n·n·E
    # Diffusion component: q·D_n·∇n
    ds.edge_model(
        device=device_name,
        region=region,
        name="ElectronCurrent",
        equation=f"q*{mu_n}*EdgeInverseLength*Electrons@n0*ElectricField_x + "
                 f"q*{D_n}*EdgeInverseLength*(Electrons@n1 - Electrons@n0)"
    )

    # Derivatives for Newton solver
    ds.edge_model(
        device=device_name,
        region=region,
        name="ElectronCurrent:Electrons@n0",
        equation=f"q*{mu_n}*EdgeInverseLength*ElectricField_x - q*{D_n}*EdgeInverseLength"
    )

    ds.edge_model(
        device=device_name,
        region=region,
        name="ElectronCurrent:Electrons@n1",
        equation=f"q*{D_n}*EdgeInverseLength"
    )

    ds.edge_model(
        device=device_name,
        region=region,
        name="ElectronCurrent:Potential@n0",
        equation=f"q*{mu_n}*EdgeInverseLength*Electrons@n0*ElectricField_x:Potential@n0"
    )

    ds.edge_model(
        device=device_name,
        region=region,
        name="ElectronCurrent:Potential@n1",
        equation=f"q*{mu_n}*EdgeInverseLength*Electrons@n0*ElectricField_x:Potential@n1"
    )

    # Hole current density (edge model)
    # Drift component: q·μ_p·p·E
    # Diffusion component: -q·D_p·∇p
    ds.edge_model(
        device=device_name,
        region=region,
        name="HoleCurrent",
        equation=f"q*{mu_p}*EdgeInverseLength*Holes@n0*ElectricField_x - "
                 f"q*{D_p}*EdgeInverseLength*(Holes@n1 - Holes@n0)"
    )

    # Derivatives for Newton solver
    ds.edge_model(
        device=device_name,
        region=region,
        name="HoleCurrent:Holes@n0",
        equation=f"q*{mu_p}*EdgeInverseLength*ElectricField_x + q*{D_p}*EdgeInverseLength"
    )

    ds.edge_model(
        device=device_name,
        region=region,
        name="HoleCurrent:Holes@n1",
        equation=f"-q*{D_p}*EdgeInverseLength"
    )

    ds.edge_model(
        device=device_name,
        region=region,
        name="HoleCurrent:Potential@n0",
        equation=f"q*{mu_p}*EdgeInverseLength*Holes@n0*ElectricField_x:Potential@n0"
    )

    ds.edge_model(
        device=device_name,
        region=region,
        name="HoleCurrent:Potential@n1",
        equation=f"q*{mu_p}*EdgeInverseLength*Holes@n0*ElectricField_x:Potential@n1"
    )

    print(f"      ✓ Drift-diffusion currents J_n and J_p created")

def build_recombination_model(device_name, region, srh=True, radiative=False, auger=False):
    """
    Build recombination rate models.

    Total recombination: U = U_SRH + U_radiative + U_Auger

    Args:
        device_name: Device identifier
        region: Region name
        srh: Enable Shockley-Read-Hall recombination
        radiative: Enable radiative recombination
        auger: Enable Auger recombination
    """
    print(f"    Building recombination models...")

    # Get parameters
    n_i = ds.get_parameter(device=device_name, name="n_i")
    tau_n = ds.get_parameter(device=device_name, region=region, name="tau_n")
    tau_p = ds.get_parameter(device=device_name, region=region, name="tau_p")

    # Initialize total recombination
    U_total = "0"

    # 1. Shockley-Read-Hall (SRH) Recombination
    if srh:
        # n1 and p1 for mid-gap trap level (E_t = E_i)
        n1 = n_i
        p1 = n_i

        # U_SRH = (n*p - n_i^2) / (tau_p*(n + n1) + tau_n*(p + p1))
        ds.node_model(
            device=device_name,
            region=region,
            name="U_SRH",
            equation=f"(Electrons*Holes - {n_i}^2) / "
                     f"({tau_p}*(Electrons + {n1}) + {tau_n}*(Holes + {p1}))"
        )

        # Derivatives for Newton solver
        ds.node_model(
            device=device_name,
            region=region,
            name="U_SRH:Electrons",
            equation=f"diff(U_SRH, Electrons)"
        )

        ds.node_model(
            device=device_name,
            region=region,
            name="U_SRH:Holes",
            equation=f"diff(U_SRH, Holes)"
        )

        U_total += " + U_SRH"
        print(f"      ✓ SRH recombination added")

    # 2. Radiative Recombination (optional)
    if radiative:
        B_rad = 1.0e-14  # Radiative recombination coefficient [cm^3/s]

        ds.node_model(
            device=device_name,
            region=region,
            name="U_radiative",
            equation=f"{B_rad}*(Electrons*Holes - {n_i}^2)"
        )

        ds.node_model(
            device=device_name,
            region=region,
            name="U_radiative:Electrons",
            equation=f"{B_rad}*Holes"
        )

        ds.node_model(
            device=device_name,
            region=region,
            name="U_radiative:Holes",
            equation=f"{B_rad}*Electrons"
        )

        U_total += " + U_radiative"
        print(f"      ✓ Radiative recombination added")

    # 3. Auger Recombination (optional)
    if auger:
        C_n = 2.8e-31  # Auger coefficient for electrons [cm^6/s]
        C_p = 9.9e-32  # Auger coefficient for holes [cm^6/s]

        ds.node_model(
            device=device_name,
            region=region,
            name="U_Auger",
            equation=f"({C_n}*Electrons + {C_p}*Holes)*(Electrons*Holes - {n_i}^2)"
        )

        ds.node_model(
            device=device_name,
            region=region,
            name="U_Auger:Electrons",
            equation=f"diff(U_Auger, Electrons)"
        )

        ds.node_model(
            device=device_name,
            region=region,
            name="U_Auger:Holes",
            equation=f"diff(U_Auger, Holes)"
        )

        U_total += " + U_Auger"
        print(f"      ✓ Auger recombination added")

    # Create total recombination model
    ds.node_model(
        device=device_name,
        region=region,
        name="Recombination",
        equation=U_total
    )

    # Derivatives
    ds.node_model(
        device=device_name,
        region=region,
        name="Recombination:Electrons",
        equation=f"diff(Recombination, Electrons)"
    )

    ds.node_model(
        device=device_name,
        region=region,
        name="Recombination:Holes",
        equation=f"diff(Recombination, Holes)"
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
        equation=f"{epsilon}*ElectricField_x"
    )

    ds.edge_model(
        device=device_name,
        region=region,
        name="ElectricFlux:Potential@n0",
        equation=f"{epsilon}*ElectricField_x:Potential@n0"
    )

    ds.edge_model(
        device=device_name,
        region=region,
        name="ElectricFlux:Potential@n1",
        equation=f"{epsilon}*ElectricField_x:Potential@n1"
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
    Construct electron continuity equation:
    (1/q)∇·J_n = U - G

    Args:
        device_name: Device identifier
        region: Region name
    """
    print(f"    Constructing electron continuity equation...")

    # Generation rate (initially zero, can be set later for illumination)
    ds.node_model(
        device=device_name,
        region=region,
        name="OpticalGeneration",
        equation="0.0"
    )

    # Net generation-recombination: R = U - G
    ds.node_model(
        device=device_name,
        region=region,
        name="ElectronNetRecombination",
        equation="Recombination - OpticalGeneration"
    )

    ds.node_model(
        device=device_name,
        region=region,
        name="ElectronNetRecombination:Electrons",
        equation="Recombination:Electrons"
    )

    ds.node_model(
        device=device_name,
        region=region,
        name="ElectronNetRecombination:Holes",
        equation="Recombination:Holes"
    )

    # Add equation to solver
    ds.equation(
        device=device_name,
        region=region,
        name="ElectronContinuity",
        variable_name="Electrons",
        edge_model="ElectronCurrent",
        edge_volume_model="",
        node_model="ElectronNetRecombination",
        variable_update="positive"  # Keep electron concentration positive
    )

    print(f"      ✓ Electron continuity: (1/q)∇·J_n = U - G")

def construct_hole_continuity_eq(device_name, region):
    """
    Construct hole continuity equation:
    -(1/q)∇·J_p = U - G

    Args:
        device_name: Device identifier
        region: Region name
    """
    print(f"    Constructing hole continuity equation...")

    # Net generation-recombination: R = U - G
    ds.node_model(
        device=device_name,
        region=region,
        name="HoleNetRecombination",
        equation="Recombination - OpticalGeneration"
    )

    ds.node_model(
        device=device_name,
        region=region,
        name="HoleNetRecombination:Electrons",
        equation="Recombination:Electrons"
    )

    ds.node_model(
        device=device_name,
        region=region,
        name="HoleNetRecombination:Holes",
        equation="Recombination:Holes"
    )

    # Add equation to solver
    ds.equation(
        device=device_name,
        region=region,
        name="HoleContinuity",
        variable_name="Holes",
        edge_model="HoleCurrent",
        edge_volume_model="",
        node_model="HoleNetRecombination",
        variable_update="positive"  # Keep hole concentration positive
    )

    print(f"      ✓ Hole continuity: -(1/q)∇·J_p = U - G")


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
        edge_charge_model="DField"
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

def set_variables(device_name):
    """
    A. Primary Solution Variables (Node Models):
         - Potential: $\psi(x, y)$
         - Electrons: $n(x, y)$
         - Holes: $p(x, y)$

    :param device_name:
    :return:
    """
    return device_name

def main():
    """Main execution function with proper argument handling."""

    # Default mesh filename - MUST match gmesh_diode2d.py output
    default_mesh = "gmsh_diode2d.msh"

    # Parse command-line arguments
    if len(sys.argv) > 2:
        print("=" * 70)
        print("ERROR: Too many arguments")
        print("=" * 70)
        print("\nUsage:")
        print(f"  python {sys.argv[0]} [mesh_file]")
        print("\nExamples:")
        print(f"  python {sys.argv[0]}                    # Uses default: {default_mesh}")
        print(f"  python {sys.argv[0]} custom_mesh.msh   # Uses specified file")
        print("=" * 70)
        sys.exit(1)

    # Use provided mesh file or default
    mesh_file = sys.argv[1] if len(sys.argv) == 2 else default_mesh

    try:
        # Setup the device
        device_name = setup_photodiode_device(mesh_file)

        print(f"\n{'=' * 70}")
        print(f"✓ SUCCESS: Device '{device_name}' is ready for equation setup!")
        print(f"{'=' * 70}\n")

    except FileNotFoundError as e:
        print(f"\n{'=' * 70}")
        print(f"✗ FILE ERROR")
        print(f"{'=' * 70}")
        print(f"\n{e}")
        print(f"\nTo generate the mesh, run:")
        print(f"  python gmesh_diode2d.py")
        print(f"{'=' * 70}\n")
        sys.exit(1)

    except ValueError as e:
        print(f"\n{'=' * 70}")
        print(f"✗ STRUCTURE ERROR")
        print(f"{'=' * 70}")
        print(f"\n{e}")
        print(f"{'=' * 70}\n")
        sys.exit(1)

    except Exception as e:
        print(f"\n{'=' * 70}")
        print(f"✗ UNEXPECTED ERROR")
        print(f"{'=' * 70}")
        print(f"\n{e}")
        print("\nFull traceback:")
        import traceback
        traceback.print_exc()
        print(f"{'=' * 70}\n")
        sys.exit(1)

    #
    device_name = setup_photodiode_device(mesh_file)
    
    add_physics(device_name)

if __name__ == "__main__":
    main()