# ============================================================================
# FILE: doping_profile.py
# PURPOSE: Define doping concentrations for p and n regions
# ============================================================================

import devsim


def define_uniform_doping(device, p_doping, n_doping):
    """
    Defines the acceptor and donor concentrations for the device.

    This function sets up:
    1. Uniform acceptor doping in p-region
    2. Uniform donor doping in n-region
    3. Net doping calculation for both regions

    The doping profile determines:
    - Built-in potential of the junction
    - Depletion region width
    - Electric field distribution

    Args:
        device (str): Name of the device
        p_doping (float): Acceptor concentration in p-region (cm^-3)
        n_doping (float): Donor concentration in n-region (cm^-3)
    """

    # ===== P-REGION DOPING =====
    # Set uniform acceptor concentration, no donors
    devsim.node_model(
        device=device,
        region="p_region",
        name="Acceptors",
        equation=f"{p_doping}"  # N_A = constant throughout p-region
    )
    devsim.node_model(
        device=device,
        region="p_region",
        name="Donors",
        equation="0.0"  # No donors in p-region
    )

    # ===== N-REGION DOPING =====
    # Set uniform donor concentration, no acceptors
    devsim.node_model(
        device=device,
        region="n_region",
        name="Acceptors",
        equation="0.0"  # No acceptors in n-region
    )
    devsim.node_model(
        device=device,
        region="n_region",
        name="Donors",
        equation=f"{n_doping}"  # N_D = constant throughout n-region
    )

    # ===== NET DOPING CALCULATION =====
    # Net doping = N_D - N_A (positive in n-region, negative in p-region)
    # This is used in space charge calculations and carrier statistics
    for region in ["p_region", "n_region"]:
        devsim.node_model(
            device=device,
            region=region,
            name="NetDoping",
            equation="Donors - Acceptors"
        )

    # Print confirmation with scientific notation
    print(f"Defined doping: N_A = {p_doping:.1e} cm^-3, N_D = {n_doping:.1e} cm^-3")
