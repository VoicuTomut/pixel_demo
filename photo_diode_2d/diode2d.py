"""
2D Photodiode DEVSIM Simulation Setup - Enhanced & Corrected Version

This script imports a Gmsh mesh and prepares the complete DEVSIM device structure
with proper geometry setup, material parameters, and verification, ready for
drift-diffusion equation implementation.


"""

import devsim as ds
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
    print("✓✓✓ DEVICE SETUP COMPLETE ✓✓✓")
    print("=" * 70)

    print(f"\nThe device '{device_name}' is fully configured and ready for simulation.")

    print("\n" + "=" * 70)
    print("📋 NEXT STEPS FOR SIMULATION:")
    print("=" * 70)
    print("""
1. DEFINE SOLUTION VARIABLES
   - Create node models: Potential, Electrons, Holes

2. CREATE MODELS FOR PHYSICS
   - Electric field (from potential gradient)
   - Electron/hole current densities (drift-diffusion)
   - Recombination models (SRH, Auger, Radiative)

3. SET UP DRIFT-DIFFUSION EQUATIONS
   - Poisson's equation (charge/potential relationship)
   - Electron continuity equation
   - Hole continuity equation

4. DEFINE BOUNDARY CONDITIONS
   - Ohmic contacts at cathode/anode
   - Surface recombination (if needed)
   - Interface continuity conditions

5. SOLVE EQUILIBRIUM (DARK, V=0)
   - Initialize variables with reasonable guesses
   - Solve coupled nonlinear system

6. PERFORM SIMULATIONS
   - DC I-V sweep (dark current)
   - Add optical generation profile
   - Calculate photocurrent and quantum efficiency
   - Spectral response analysis
""")
    print("=" * 70)

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


if __name__ == "__main__":
    main()