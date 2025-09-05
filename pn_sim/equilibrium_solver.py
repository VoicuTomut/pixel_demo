# pn_sim/equilibrium_solver.py

import devsim
import numpy as np
from pn_sim.physics_setup import setup_carrier_transport_equations


def solve_initial_equilibrium(device):
    """
    Solves for equilibrium using a robust multi-step convergence strategy
    with gradually tightening tolerances. This is a standard, version-agnostic
    method for achieving stable convergence.
    """
    print("\n--- Solving for equilibrium (Multi-Step Convergence Method) ---")

    # --- STAGE 1: SET A SELF-CONSISTENT INITIAL GUESS ---
    print("  Stage 1: Setting a physically consistent initial guess...")
    V_t = devsim.get_parameter(name="ThermalVoltage")
    for region in ["p_region", "n_region"]:
        ni_vals = np.array(devsim.get_node_model_values(device=device, region=region, name="IntrinsicCarrierDensity"))
        net_doping = np.array(devsim.get_node_model_values(device=device, region=region, name="NetDoping"))

        n_eq = 0.5 * (net_doping + np.sqrt(net_doping ** 2 + 4 * ni_vals ** 2))
        p_eq = 0.5 * (-net_doping + np.sqrt(net_doping ** 2 + 4 * ni_vals ** 2))
        devsim.set_node_values(device=device, region=region, name="Electrons", values=np.maximum(n_eq, 1.0).tolist())
        devsim.set_node_values(device=device, region=region, name="Holes", values=np.maximum(p_eq, 1.0).tolist())

        psi_initial = V_t * np.arcsinh(net_doping / (2.0 * ni_vals))
        devsim.set_node_values(device=device, region=region, name="Potential", values=psi_initial.tolist())
    print("  ✓ Self-consistent initial guess is set.")

    # --- STAGE 2: DEFINE ALL EQUATIONS ONCE ---
    # This ensures a complete and consistent system definition from the start.
    setup_carrier_transport_equations(device)

    # --- STAGE 3: SOLVE WITH GRADUALLY TIGHTENING TOLERANCES ---
    print("\n  Stage 3: Solving the fully coupled system...")
    try:
        # Step 3a: Loose solve to get close to the solution
        print("    - Step 3a: Solving with loose tolerance...")
        devsim.solve(type="dc", absolute_error=10, relative_error=1e-1, maximum_iterations=60)

        # Step 3b: Medium solve for better refinement
        print("    - Step 3b: Solving with medium tolerance...")
        devsim.solve(type="dc", absolute_error=10, relative_error=5e-2, maximum_iterations=200)

        # Step 3c: Final, tight solve for high accuracy
        print("    - Step 3c: Solving with tight tolerance...")
        devsim.solve(type="dc", absolute_error=10.0, relative_error=5e-3, maximum_iterations=200)

        print("\n  ✓ Coupled system converged successfully.")

    except devsim.error as e:
        print(f"  ❌ FATAL: Convergence failed during coupled solve: {e}")
        debug_equilibrium_state(device)
        raise

    # --- STAGE 4: VERIFY THE SOLUTION ---
    verify_equilibrium(device)


def verify_equilibrium(device):
    """Verifies that a physically correct equilibrium was reached."""
    print("\n  --- Verifying Equilibrium Solution ---")
    try:
        psi_p_nodes = devsim.get_node_model_values(device=device, region="p_region", name="Potential")
        psi_n_nodes = devsim.get_node_model_values(device=device, region="n_region", name="Potential")

        # Use a robust measure away from the junction
        psi_p = np.median(psi_p_nodes)
        psi_n = np.median(psi_n_nodes)
        v_bi = psi_n - psi_p

        print(f"    Built-in potential (V_bi): {v_bi:.4f} V")
        if 0.5 < v_bi < 0.9:
            print("    ✅ SUCCESS: Built-in potential is physically correct.")
        else:
            print(f"    ⚠️ WARNING: Built-in potential ({v_bi:.4f} V) is outside the expected range.")
    except devsim.error as e:
        print(f"    ❌ Could not verify V_bi: {e}")


def debug_equilibrium_state(device):
    """Provides a detailed debug output of the device state."""
    print("\n" + "=" * 50)
    print("EQUILIBRIUM STATE DEBUG")
    print("=" * 50)
    for region in ["p_region", "n_region"]:
        try:
            potential = np.array(devsim.get_node_model_values(device=device, region=region, name="Potential"))
            electrons = np.array(devsim.get_node_model_values(device=device, region=region, name="Electrons"))
            holes = np.array(devsim.get_node_model_values(device=device, region=region, name="Holes"))
            print(f"\n{region}:\n  Potential Range: [{np.min(potential):.3f}, {np.max(potential):.3f}] V")
            print(f"  Electron Range:  [{np.min(electrons):.2e}, {np.max(electrons):.2e}] cm⁻³")
            print(f"  Hole Range:      [{np.min(holes):.2e}, {np.max(holes):.2e}] cm⁻³")
        except Exception as e:
            print(f"  Could not get values for {region}: {e}")
    print("=" * 50 + "\n")