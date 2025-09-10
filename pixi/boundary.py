# ============================================================================
# FILE: boundary.py
# PURPOSE: Define boundary conditions for the photodiode device
# ============================================================================

import devsim


def set_boundary(device_name):
    """
    Sets up boundary conditions for the photodiode device.

    This function establishes:
    1. Contact boundary conditions at anode and cathode
    2. Interface continuity conditions at the p-n junction

    Args:
        device_name (str): Name of the device in DEVSIM
    """
    print("  3C: Defining all boundary condition models...")

    # ===== CONTACT BOUNDARY CONDITIONS =====
    # Define boundary conditions for both metal contacts (anode and cathode)
    for contact in ["anode", "cathode"]:
        # Initialize bias voltage parameter for this contact (starts at 0V)
        devsim.set_parameter(device=device_name, name=f"{contact}_bias", value=0.0)

        # --- Potential Boundary Condition ---
        # Enforces: Potential at contact = applied bias voltage
        # This creates a Dirichlet boundary condition for the Poisson equation
        devsim.contact_node_model(
            device=device_name,
            contact=contact,
            name=f"{contact}_potential_bc",
            equation=f"Potential - {contact}_bias"  # BC: V = V_applied
        )
        # Derivative with respect to Potential variable (needed for Newton solver)
        devsim.contact_node_model(
            device=device_name,
            contact=contact,
            name=f"{contact}_potential_bc:Potential",
            equation="1.0"  # d(BC)/d(Potential) = 1
        )

        # --- Electron Boundary Condition ---
        # Enforces charge neutrality at contact: n = n_i (intrinsic concentration)
        # This assumes ohmic contact behavior
        devsim.contact_node_model(
            device=device_name,
            contact=contact,
            name=f"{contact}_electrons_bc",
            equation="Electrons - IntrinsicElectrons"  # BC: n = n_intrinsic
        )
        # Derivative with respect to Electrons variable
        devsim.contact_node_model(
            device=device_name,
            contact=contact,
            name=f"{contact}_electrons_bc:Electrons",
            equation="1.0"  # d(BC)/d(Electrons) = 1
        )

        # --- Hole Boundary Condition ---
        # Enforces charge neutrality at contact: p = p_i (intrinsic concentration)
        # Complements the electron BC for overall charge balance
        devsim.contact_node_model(
            device=device_name,
            contact=contact,
            name=f"{contact}_holes_bc",
            equation="Holes - IntrinsicHoles"  # BC: p = p_intrinsic
        )
        # Derivative with respect to Holes variable
        devsim.contact_node_model(
            device=device_name,
            contact=contact,
            name=f"{contact}_holes_bc:Holes",
            equation="1.0"  # d(BC)/d(Holes) = 1
        )

    # ===== INTERFACE CONTINUITY CONDITIONS =====
    # At the p-n junction interface, ensure continuous variables across regions
    # This maintains physical consistency at the junction boundary

    for variable in ["Potential", "Electrons", "Holes"]:
        # Continuity condition: Variable must be equal on both sides of interface
        # r0 = p_region side, r1 = n_region side
        devsim.interface_model(
            device=device_name,
            interface="pn_junction",
            name=f"{variable}_continuity",
            equation=f"{variable}@r0 - {variable}@r1"  # Continuity: V_p = V_n
        )

        # Derivative with respect to variable on p-region side (r0)
        devsim.interface_model(
            device=device_name,
            interface="pn_junction",
            name=f"{variable}_continuity:{variable}@r0",
            equation="1.0"  # d(continuity)/d(V@r0) = 1
        )

        # Derivative with respect to variable on n-region side (r1)
        devsim.interface_model(
            device=device_name,
            interface="pn_junction",
            name=f"{variable}_continuity:{variable}@r1",
            equation="-1.0"  # d(continuity)/d(V@r1) = -1
        )
