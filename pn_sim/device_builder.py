# pn_sim/device_builder.py
"""
Handles mesh loading, device creation, and geometry verification.
"""
import devsim
import os
import numpy as np
from .utils import get_alpha_for_wavelength


def load_mesh_and_create_device(device_name, mesh_file):
    """Loads the GMSH mesh file and creates the DEVSIM device structure."""
    print(f"Attempting to load mesh: '{mesh_file}'")
    if not os.path.exists(mesh_file):
        raise FileNotFoundError(f"Mesh file '{mesh_file}' not found. Please run create_mesh.py first.")

    devsim.create_gmsh_mesh(mesh=device_name, file=mesh_file)
    devsim.add_gmsh_region(mesh=device_name, gmsh_name="p_region", region="p_region", material="Silicon")
    devsim.add_gmsh_region(mesh=device_name, gmsh_name="n_region", region="n_region", material="Silicon")
    # FIX: Added the required 'material="metal"' parameter to both contact definitions.
    devsim.add_gmsh_contact(mesh=device_name, gmsh_name="anode", region="p_region", name="anode", material="metal")
    devsim.add_gmsh_contact(mesh=device_name, gmsh_name="cathode", region="n_region", name="cathode", material="metal")
    # FIX: Added the required 'material="metal"' parameter to both contact definitions.

    devsim.add_gmsh_interface(mesh=device_name, gmsh_name="pn_junction", region0="p_region", region1="n_region",
                              name="pn_junction")
    devsim.finalize_mesh(mesh=device_name)
    devsim.create_device(mesh=device_name, device=device_name)
    print("--- Mesh loaded and device created successfully ---")


def verify_device_structure(device_name):
    """Performs a quick check to ensure the device was created with the expected components."""
    print("\n--- Verifying Device Structure ---")
    try:
        dev_list = devsim.get_device_list()
        reg_list = devsim.get_region_list(device=device_name)
        con_list = devsim.get_contact_list(device=device_name)
        int_list = devsim.get_interface_list(device=device_name)

        assert len(dev_list) == 1, f"Expected 1 device, found {len(dev_list)}"
        assert len(reg_list) == 2, f"Expected 2 regions, found {len(reg_list)}"
        assert len(con_list) == 2, f"Expected 2 contacts, found {len(con_list)}"
        assert len(int_list) == 1, f"Expected 1 interface, found {len(int_list)}"
        print("✅ Verification PASSED: Device structure (regions, contacts, interface) is correct.")
    except (devsim.error, AssertionError) as e:
        print(f"❌ Verification FAILED: {e}")
        raise


def comprehensive_mesh_debug(device, material_params):
    """Analyzes mesh geometry and its implications for photodiode performance."""
    print("\n--- Detailed Mesh and Geometry Analysis ---")
    all_x, all_y = [], []
    for region in ["p_region", "n_region"]:
        x = np.array(devsim.get_node_model_values(device=device, region=region, name="x"))
        y = np.array(devsim.get_node_model_values(device=device, region=region, name="y"))
        all_x.extend(x)
        all_y.extend(y)
        print(f"  Region '{region}': {len(x)} nodes, y-range [{np.min(y):.2f}, {np.max(y):.2f}] μm")

    device_depth = abs(np.min(all_y))
    print(f"  Total Device Depth: {device_depth:.2f} μm")

    # Assess optical absorption capability
    wavelengths_to_check = [450, 650, 850]  # Blue, Red, Near-IR
    print("  Optical Absorption Potential:")
    for wl in wavelengths_to_check:
        alpha = get_alpha_for_wavelength(wl, material_params)
        absorption_length = 1e4 / alpha  # in um
        absorbed_fraction = 1 - np.exp(-device_depth / absorption_length)
        print(f"    - At {wl} nm (1/α = {absorption_length:.2f} μm): {absorbed_fraction:.1%} of light is absorbed.")

    if device_depth < 5.0:
        print("  ⚠️ WARNING: Device depth is shallow, may have low quantum efficiency for longer wavelengths (>800nm).")
    else:
        print("  ✅ Assessment: Device depth is sufficient for good visible light absorption.")

