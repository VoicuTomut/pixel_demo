# ============================================================================
# FILE: solve_equilibrium.py
# PURPOSE: Solve for initial equilibrium state of the device
# ============================================================================

import devsim


def solve_equilibrium(device_name):
    """
    Solves for the initial equilibrium state using a two-step method.

    This robust approach:
    1. First solves Poisson equation alone for potential
    2. Updates carrier concentrations using Boltzmann statistics
    3. Then solves the full coupled system

    This method provides better convergence than solving all equations
    simultaneously from a poor initial guess.

    Args:
        device_name (str): Name of the device
    """
    print("  3D: Solving for initial equilibrium state (two-step method)...")

    # ===== STEP 1: CREATE INITIAL GUESS =====
    print("    Creating robust initial guess based on charge neutrality...")

    for region in ["p_region", "n_region"]:
        # Initialize carriers at intrinsic values (charge neutral)
        devsim.set_node_values(
            device=device_name, region=region,
            name="Electrons", init_from="IntrinsicElectrons"
        )
        devsim.set_node_values(
            device=device_name, region=region,
            name="Holes", init_from="IntrinsicHoles"
        )

        # Calculate initial potential from quasi-Fermi levels
        # φ = V_t * ln(n/n_i) for n-type regions
        devsim.node_model(
            device=device_name, region=region, name="InitialPotential",
            equation="ThermalVoltage * log(IntrinsicElectrons/IntrinsicCarrierDensity)"
        )
        devsim.set_node_values(
            device=device_name, region=region,
            name="Potential", init_from="InitialPotential"
        )

    # ===== STEP 2: SOLVE POISSON EQUATION ONLY =====
    print("    Step 1/2: Assembling and solving for Potential only...")

    # Setup Poisson equation in both regions
    # ∇·(ε∇φ) = -ρ where ρ = q(p - n + N_D - N_A)
    for region in ["p_region", "n_region"]:
        devsim.equation(
            device=device_name, region=region,
            name="PotentialEquation",
            variable_name="Potential",
            node_model="SpaceCharge",  # Source term: -ρ/ε
            edge_model="DField",  # Flux term: D = ε∇φ
            variable_update="log_damp"  # Use logarithmic damping for stability
        )

    # Setup contact boundary conditions for Potential
    for contact in ["anode", "cathode"]:
        devsim.contact_equation(
            device=device_name, contact=contact,
            name="PotentialEquation",
            node_model=f"{contact}_potential_bc"  # V = V_applied at contact
        )

    # Setup interface continuity for Potential
    devsim.interface_equation(
        device=device_name, interface="pn_junction",
        name="PotentialEquation",
        interface_model="Potential_continuity",
        type="continuous"  # Potential is continuous across junction
    )

    # Solve Poisson equation with tight tolerances
    devsim.solve(
        type="dc",
        absolute_error=1e-10,
        relative_error=1e-12,
        maximum_iterations=50
    )

    # ===== STEP 3: UPDATE CARRIER CONCENTRATIONS =====
    print("    Updating carrier guess using Boltzmann statistics...")

    for region in ["p_region", "n_region"]:
        # Update electrons: n = n_i * exp(φ/V_t)
        devsim.node_model(
            device=device_name, region=region, name="UpdatedElectrons",
            equation="IntrinsicCarrierDensity*exp(Potential/ThermalVoltage)"
        )

        # Update holes: p = n_i * exp(-φ/V_t)
        devsim.node_model(
            device=device_name, region=region, name="UpdatedHoles",
            equation="IntrinsicCarrierDensity*exp(-Potential/ThermalVoltage)"
        )

        # Apply updated values
        devsim.set_node_values(
            device=device_name, region=region,
            name="Electrons", init_from="UpdatedElectrons"
        )
        devsim.set_node_values(
            device=device_name, region=region,
            name="Holes", init_from="UpdatedHoles"
        )

    # ===== STEP 4: SOLVE FULL COUPLED SYSTEM =====
    print("    Step 2/2: Assembling continuity equations and solving the full system...")

    # Setup electron and hole continuity equations in regions
    for region in ["p_region", "n_region"]:
        # Electron continuity: ∂n/∂t + ∇·J_n = -q(R-G)
        # In steady state: ∇·J_n = q(R-G)
        devsim.equation(
            device=device_name, region=region,
            name="ElectronContinuityEquation",
            variable_name="Electrons",
            node_model="eCharge_x_NetRecomb",  # Source: q(R-G)
            edge_model="ElectronCurrent",  # Flux: J_n
            variable_update="log_damp"  # Log damping for positivity
        )

        # Hole continuity: ∂p/∂t + ∇·J_p = q(R-G)
        # In steady state: ∇·J_p = -q(R-G)
        devsim.equation(
            device=device_name, region=region,
            name="HoleContinuityEquation",
            variable_name="Holes",
            node_model="Neg_eCharge_x_NetRecomb",  # Source: -q(R-G)
            edge_model="HoleCurrent",  # Flux: J_p
            variable_update="log_damp"  # Log damping for positivity
        )

    # Setup interface continuity for carriers
    devsim.interface_equation(
        device=device_name, interface="pn_junction",
        name="ElectronContinuityEquation",
        interface_model="Electrons_continuity",
        type="continuous"
    )
    devsim.interface_equation(
        device=device_name, interface="pn_junction",
        name="HoleContinuityEquation",
        interface_model="Holes_continuity",
        type="continuous"
    )

    # ===== COMPLETE CONTACT EQUATIONS =====
    # Add all contact boundary conditions with proper edge models
    for contact in ["anode", "cathode"]:
        # Potential BC with edge charge model (critical for C-V)
        devsim.contact_equation(
            device=device_name, contact=contact,
            name="PotentialEquation",
            node_model=f"{contact}_potential_bc",
            edge_charge_model="DField"  # Include displacement current
        )

        # Electron continuity BC with current model
        devsim.contact_equation(
            device=device_name, contact=contact,
            name="ElectronContinuityEquation",
            node_model=f"{contact}_electrons_bc",
            edge_current_model="ElectronCurrent"
        )

        # Hole continuity BC with current model
        devsim.contact_equation(
            device=device_name, contact=contact,
            name="HoleContinuityEquation",
            node_model=f"{contact}_holes_bc",
            edge_current_model="HoleCurrent"
        )
