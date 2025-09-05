# pn_sim/equilibrium_solver.py
"""
Functions for solving the device at thermal equilibrium and debugging the state.
"""
import devsim
import numpy as np


def solve_initial_equilibrium(device):
    """
    Solves for the initial equilibrium state (0V bias, no light).
    This provides a stable starting point for further simulations.
    Strategy: First solve Poisson's equation only, then the full coupled system.
    """
    print("\n--- Solving for Equilibrium (0V, Dark) ---")

    devsim.set_parameter(name="anode_bias", value=0.0)
    devsim.set_parameter(name="cathode_bias", value=0.0)

    # Solve Poisson's equation alone for a good initial guess of the potential distribution.
    for region in ["p_region", "n_region"]:
        devsim.disable_equation(device=device, region=region, name="ElectronContinuityEquation")
        devsim.disable_equation(device=device, region=region, name="HoleContinuityEquation")

    print("  Solving Poisson's equation alone...")
    devsim.solve(type="dc", absolute_error=1.0, relative_error=1e-12, maximum_iterations=30)

    # Re-enable continuity equations and solve the fully coupled drift-diffusion system.
    for region in ["p_region", "n_region"]:
        devsim.enable_equation(device=device, region=region, name="ElectronContinuityEquation")
        devsim.enable_equation(device=device, region=region, name="HoleContinuityEquation")

    print("  Solving fully coupled system for equilibrium...")
    devsim.solve(type="dc", absolute_error=1e10, relative_error=1e-10, maximum_iterations=30)

    devsim.write_devices(file="output/equilibrium_solution.tec", type="tecplot")
    print("  Equilibrium solution found and saved to 'output/equilibrium_solution.tec'")


def debug_equilibrium_state(device, material_params):
    """
    Analyzes the solved equilibrium state and compares key metrics
    to theoretical values for validation.
    """
    print("\n--- Debugging Equilibrium State ---")

    # Extract doping levels from parameters
    Na = material_params["peak_p_doping"]
    Nd = material_params["n_bulk_doping"]
    ni = material_params["n_i"]
    vt = 0.02585  # Thermal voltage at 300K in V

    # --- Built-in Potential (Vbi) ---
    # Formula: Vbi = (kT/q) * ln(Na*Nd / ni^2)
    vbi_theory = vt * np.log(Na * Nd / ni ** 2)

    # Get simulated potential drop
    v_anode = devsim.get_parameter(device=device, name="anode_bias")
    v_cathode = devsim.get_parameter(device=device, name="cathode_bias")

    potential_p = np.array(devsim.get_node_model_values(device=device, region="p_region", name="Potential"))
    potential_n = np.array(devsim.get_node_model_values(device=device, region="n_region", name="Potential"))

    vbi_sim = np.max(potential_n) - np.min(potential_p)

    print(f"  Built-in Potential (Vbi):")
    print(f"    - Theoretical: {vbi_theory:.4f} V")
    print(f"    - Simulated:   {vbi_sim:.4f} V")

    diff_percent = 100 * abs(vbi_sim - vbi_theory) / vbi_theory
    if diff_percent > 10:
        print(f"    - ⚠️ WARNING: High deviation ({diff_percent:.1f}%) from theory.")
    else:
        print(f"    - ✅ Assessment: Good agreement with theory ({diff_percent:.1f}% difference).")

    # --- Depletion Width (W) ---
    # Formula: W = sqrt( (2*eps/q) * (1/Na + 1/Nd) * Vbi )
    eps = material_params["permittivity"]
    q = material_params["electron_charge"]

    # CLARIFICATION: The theoretical formula below is for an ideal abrupt junction.
    # The simulated device uses a graded (Gaussian) junction, so some deviation is expected.
    w_theory = np.sqrt(2 * eps / q * (1 / Na + 1 / Nd) * vbi_theory) * 1e4  # convert cm to um

    # Get simulated depletion region from SpaceCharge
    sc_p = np.array(devsim.get_node_model_values(device=device, region="p_region", name="SpaceCharge"))
    y_p = np.array(devsim.get_node_model_values(device=device, region="p_region", name="y"))
    depleted_y_p = y_p[abs(sc_p) > q * Na * 0.1]

    sc_n = np.array(devsim.get_node_model_values(device=device, region="n_region", name="SpaceCharge"))
    y_n = np.array(devsim.get_node_model_values(device=device, region="n_region", name="y"))
    depleted_y_n = y_n[abs(sc_n) > q * Nd * 0.1]

    if len(depleted_y_p) > 0 and len(depleted_y_n) > 0:
        w_sim = np.max(depleted_y_p) - np.min(depleted_y_n)
        print(f"  Depletion Width (W):")
        print(f"    - Theoretical (Abrupt Approx.): {w_theory:.4f} μm")
        print(f"    - Simulated (Graded):           {w_sim:.4f} μm")
        diff_percent_w = 100 * abs(w_sim - w_theory) / w_theory
        if diff_percent_w > 20:
            print(
                f"    - ⚠️ WARNING: High deviation ({diff_percent_w:.1f}%) from theory (as expected for graded junction).")
        else:
            print(f"    - ✅ Assessment: Reasonable agreement with theory.")
    else:
        print("  Could not estimate simulated depletion width.")