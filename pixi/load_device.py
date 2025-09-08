# load_device.py
"""

"""

import os
import devsim


def initialize_device_and_mesh(device_name, mesh_file):
    """
    Step 1: Initialize device and load mesh from GMSH file.

    Args:
        device_name (str): Name of the device to create
        mesh_file (str): Path to the GMSH mesh file

    Returns:
        bool: True if successful, False otherwise

    Raises:
        FileNotFoundError: If mesh file doesn't exist
        devsim.error: If there's an error in mesh/device creation
    """
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

###################################
def debug_device_and_mesh(device_name, mesh_file=None, verbose=True):
    """
    Simple debug function to check mesh dimensions and basic device structure.
    Call this AFTER running initialize_device_and_mesh().

    Args:
        device_name (str): Name of the device to debug
        mesh_file (str, optional): Path to the original mesh file
        verbose (bool): Enable detailed output

    Returns:
        dict: Debug information with mesh dimensions and structure
    """
    debug_info = {
        'device_exists': False,
        'regions': {},
        'contacts': {},
        'interfaces': {},
        'mesh_dimensions': {},
        'errors': []
    }

    def debug_print(message):
        if verbose:
            print(f"  {message}")

    if verbose:
        print("=" * 50)
        print("        MESH DEBUG INFO")
        print("=" * 50)

    try:
        # Check if device exists
        device_list = devsim.get_device_list()
        if device_name not in device_list:
            debug_info['errors'].append(f"Device '{device_name}' not found")
            return debug_info

        debug_info['device_exists'] = True
        debug_print(f"Device found: {device_name}")

        # Get basic structure
        regions = devsim.get_region_list(device=device_name)
        contacts = devsim.get_contact_list(device=device_name)
        interfaces = devsim.get_interface_list(device=device_name)

        debug_print(f"Regions: {list(regions)}")
        debug_print(f"Contacts: {list(contacts)}")
        debug_print(f"Interfaces: {list(interfaces)}")

        # Analyze each region
        for region in regions:
            region_info = {'name': region}

            try:
                # Get coordinates
                x_coords = devsim.get_node_model_values(device=device_name, region=region, name="x")
                y_coords = devsim.get_node_model_values(device=device_name, region=region, name="y")

                if len(x_coords) > 0:
                    x_min, x_max = min(x_coords), max(x_coords)
                    y_min, y_max = min(y_coords), max(y_coords)
                    width = x_max - x_min
                    height = abs(y_max - y_min)  # abs for depth

                    region_info.update({
                        'nodes': len(x_coords),
                        'x_range': [x_min, x_max],
                        'y_range': [y_min, y_max],
                        'width_um': width,
                        'height_um': height
                    })

                    debug_print(f"Region '{region}':")
                    debug_print(f"  Nodes: {len(x_coords)}")
                    debug_print(f"  Dimensions: {width:.3f} x {height:.3f} um")
                    debug_print(f"  X range: [{x_min:.3f}, {x_max:.3f}]")
                    debug_print(f"  Y range: [{y_min:.3f}, {y_max:.3f}]")
                else:
                    region_info['nodes'] = 0
                    debug_print(f"Region '{region}': NO NODES")

            except Exception as e:
                region_info['error'] = str(e)
                debug_print(f"Region '{region}': Error accessing coordinates - {e}")

            debug_info['regions'][region] = region_info

        # Get overall mesh dimensions
        if debug_info['regions']:
            all_x = []
            all_y = []
            total_nodes = 0

            for region_info in debug_info['regions'].values():
                if 'x_range' in region_info:
                    all_x.extend(region_info['x_range'])
                    all_y.extend(region_info['y_range'])
                    total_nodes += region_info.get('nodes', 0)

            if all_x and all_y:
                overall_width = max(all_x) - min(all_x)
                overall_height = abs(max(all_y) - min(all_y))

                debug_info['mesh_dimensions'] = {
                    'total_nodes': total_nodes,
                    'overall_width_um': overall_width,
                    'overall_height_um': overall_height,
                    'x_bounds': [min(all_x), max(all_x)],
                    'y_bounds': [min(all_y), max(all_y)]
                }

                debug_print("")
                debug_print("OVERALL MESH:")
                debug_print(f"  Total nodes: {total_nodes}")
                debug_print(f"  Overall size: {overall_width:.3f} x {overall_height:.3f} um")
                debug_print(f"  Expected size: ~5.0 x 5.0 um")

                # Check if dimensions are reasonable
                if overall_width < 1.0 or overall_height < 1.0:
                    debug_info['errors'].append("Mesh dimensions are too small")
                    debug_print("  WARNING: Mesh is very small!")
                elif overall_width > 50.0 or overall_height > 50.0:
                    debug_info['errors'].append("Mesh dimensions are too large")
                    debug_print("  WARNING: Mesh is very large!")
                else:
                    debug_print("  Dimensions look reasonable")

        # Check mesh file size if provided
        if mesh_file and os.path.exists(mesh_file):
            file_size = os.path.getsize(mesh_file)
            debug_print(f"Mesh file size: {file_size} bytes")
            debug_info['mesh_file_size'] = file_size

        # Store contact and interface info
        debug_info['contacts'] = {contact: {'name': contact} for contact in contacts}
        debug_info['interfaces'] = {interface: {'name': interface} for interface in interfaces}

    except Exception as e:
        debug_info['errors'].append(f"Debug failed: {e}")
        debug_print(f"ERROR: {e}")

    if verbose:
        print("=" * 50)
        if debug_info['errors']:
            print("ERRORS FOUND:")
            for error in debug_info['errors']:
                print(f"  - {error}")
        else:
            print("DEBUG COMPLETE - No major issues detected")
        print("=" * 50)

    return debug_info


def print_debug_summary(debug_info):
    """Print a quick summary of mesh info."""
    print("\nMESH SUMMARY:")
    print("-" * 30)

    if debug_info.get('mesh_dimensions'):
        dims = debug_info['mesh_dimensions']
        print(f"Total nodes: {dims['total_nodes']}")
        print(f"Size: {dims['overall_width_um']:.3f} x {dims['overall_height_um']:.3f} um")

    print(f"Regions: {len(debug_info.get('regions', {}))}")
    print(f"Contacts: {len(debug_info.get('contacts', {}))}")
    print(f"Interfaces: {len(debug_info.get('interfaces', {}))}")

    if debug_info.get('errors'):
        print(f"Errors: {len(debug_info['errors'])}")
        for error in debug_info['errors']:
            print(f"  - {error}")
    else:
        print("Status: OK")
    print("-" * 30)