# ============================================================================
# FILE: set_material.py
# PURPOSE: Set material parameters for semiconductor regions
# ============================================================================

import devsim


def set_material_parameters(device, region, material_descriptor):
    """
    Sets the basic material parameters for Silicon.

    These parameters define the fundamental properties of the semiconductor:
    - Permittivity: Determines electric field distribution
    - Intrinsic carrier density: Sets equilibrium carrier concentrations
    - Carrier lifetimes: Control recombination rates

    Args:
        device (str): Name of the device
        region (str): Region name ("p_region" or "n_region")
        material_descriptor (dict): Dictionary containing material properties
                                   with 'value' and 'unit' fields
    """

    # ===== DIELECTRIC PERMITTIVITY =====
    # ε = ε_0 * ε_r where ε_r ≈ 11.7 for Silicon
    devsim.set_parameter(
        device=device,
        region=region,
        name="Permittivity",
        value=material_descriptor["Permittivity"]["value"]
    )

    # ===== INTRINSIC CARRIER DENSITY =====
    # n_i ≈ 1.45e10 cm^-3 for Silicon at 300K
    # Determines equilibrium carrier concentrations
    devsim.set_parameter(
        device=device,
        region=region,
        name="IntrinsicCarrierDensity",
        value=material_descriptor["IntrinsicCarrierDensity"]["value"]
    )

    # ===== ELEMENTARY CHARGE =====
    # q = 1.602e-19 Coulombs
    devsim.set_parameter(
        device=device,
        region=region,
        name="ElectronCharge",
        value=material_descriptor["ElectronCharge"]["value"]
    )

    # ===== CARRIER LIFETIMES =====
    # τ_n and τ_p determine SRH recombination rates
    # Typical values: 1e-6 to 1e-3 seconds for Silicon
    devsim.set_parameter(
        device=device,
        region=region,
        name="taun",  # Electron lifetime
        value=material_descriptor["Tau"]["value"]
    )
    devsim.set_parameter(
        device=device,
        region=region,
        name="taup",  # Hole lifetime
        value=material_descriptor["Tau"]["value"]
    )
