# ============================================================================
# FILE: mobility_model.py
# PURPOSE: Define carrier mobility models with doping dependence
# ============================================================================

import devsim


def define_mobility_models(device, region):
    """
    Defines doping-dependent mobility using the Caughey-Thomas model.

    The Caughey-Thomas model captures how carrier mobility decreases
    with increasing doping concentration due to:
    - Ionized impurity scattering
    - Carrier-carrier scattering

    Model equation:
    μ = μ_min + (μ_max - μ_min) / (1 + (N_total/N_ref)^α)

    Args:
        device (str): Name of the device
        region (str): Region name ("p_region" or "n_region")
    """

    # ===== CALCULATE TOTAL DOPING =====
    # Total ionized impurity concentration for scattering calculations
    devsim.node_model(
        device=device,
        region=region,
        name="TotalDoping",
        equation="abs(Acceptors) + abs(Donors)"  # Total impurity concentration
    )

    # ===== ELECTRON MOBILITY MODEL =====
    # Caughey-Thomas parameters for electrons in Silicon
    mu_max_n = 1417.0  # Maximum mobility at low doping (cm²/V·s)
    mu_min_n = 68.5  # Minimum mobility at high doping (cm²/V·s)
    N_ref_n = 1.10e17  # Reference doping concentration (cm⁻³)
    alpha_n = 0.711  # Fitting parameter (dimensionless)

    # Build electron mobility equation
    eqn_n = (f"{mu_min_n} + ({mu_max_n} - {mu_min_n}) / "
             f"(1 + (TotalDoping / {N_ref_n})^{alpha_n})")

    devsim.node_model(
        device=device,
        region=region,
        name="ElectronMobility",
        equation=eqn_n
    )

    # ===== HOLE MOBILITY MODEL =====
    # Caughey-Thomas parameters for holes in Silicon
    mu_max_p = 470.5  # Maximum mobility at low doping (cm²/V·s)
    mu_min_p = 44.9  # Minimum mobility at high doping (cm²/V·s)
    N_ref_p = 2.23e17  # Reference doping concentration (cm⁻³)
    alpha_p = 0.719  # Fitting parameter (dimensionless)

    # Build hole mobility equation
    eqn_p = (f"{mu_min_p} + ({mu_max_p} - {mu_min_p}) / "
             f"(1 + (TotalDoping / {N_ref_p})^{alpha_p})")

    devsim.node_model(
        device=device,
        region=region,
        name="HoleMobility",
        equation=eqn_p
    )
