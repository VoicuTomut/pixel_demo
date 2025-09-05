import devsim
from .utils import get_alpha_for_wavelength, get_reflectivity
import numpy as np


def setup_physics_and_materials(device, material_params, global_params):
    """
    Sets up material parameters and defines all physical models and equations.
    This includes doping, Poisson's equation, drift-diffusion, SRH recombination,
    and optical generation.
    """
    print("\n--- Setting Up Physics and Material Parameters ---")

    mu_scale = 1e-4
    k_boltzmann = 8.617333262e-5
    vt = k_boltzmann * global_params["temperature_K"]

    devsim.set_parameter(name="Vt", value=vt)
    devsim.set_parameter(name="n_i", value=material_params["n_i"])
    devsim.set_parameter(name="Permittivity", value=material_params["permittivity"])
    devsim.set_parameter(name="ElectronCharge", value=material_params["electron_charge"])
    devsim.set_parameter(name="mu_n", value=material_params["mu_n_low_field"] * mu_scale)
    devsim.set_parameter(name="mu_p", value=material_params["mu_p_low_field"] * mu_scale)
    devsim.set_parameter(name="tau_n", value=material_params["tau_n"])
    devsim.set_parameter(name="tau_p", value=material_params["tau_p"])

    print("  Material parameters loaded.")

    for region in ["p_region", "n_region"]:
        # --- 1. Define Solution Variables (MUST BE FIRST) ---
        devsim.node_solution(device=device, region=region, name="Potential")
        devsim.node_solution(device=device, region=region, name="Electrons")
        devsim.node_solution(device=device, region=region, name="Holes")

        devsim.edge_from_node_model(device=device, region=region, node_model="Potential")
        devsim.edge_from_node_model(device=device, region=region, node_model="Electrons")
        devsim.edge_from_node_model(device=device, region=region, node_model="Holes")

        # --- 2. Define ALL Physical Models ---
        devsim.node_model(device=device, region=region, name="N_A",
                          equation=f"{material_params['peak_p_doping']}*exp(-y^2/(2*0.1^2))")
        devsim.node_model(device=device, region=region, name="N_D",
                          equation=f"{material_params['n_bulk_doping']}")
        devsim.node_model(device=device, region=region, name="NetDoping",
                          equation="N_D - N_A")

        devsim.node_model(device=device, region=region, name="n_eq",
                          equation="0.5*(NetDoping + (NetDoping^2 + 4*n_i^2))")
        devsim.node_model(device=device, region=region, name="p_eq",
                          equation="0.5*(-NetDoping + (NetDoping^2 + 4*n_i^2))")
        devsim.set_node_values(device=device, region=region, name="Electrons", init_from="n_eq")
        devsim.set_node_values(device=device, region=region, name="Holes", init_from="p_eq")

        devsim.node_model(device=device, region=region, name="SpaceCharge",
                          equation="ElectronCharge * (Holes - Electrons + NetDoping)")

        devsim.node_model(device=device, region=region, name="NegativeSpaceCharge",
                          equation="-SpaceCharge")

        srh_eq = "(Electrons*Holes - n_i^2)/(tau_p*(Electrons + n_i) + tau_n*(Holes + n_i))"
        devsim.node_model(device=device, region=region, name="SRHRecombination", equation=srh_eq)

        devsim.set_parameter(name="EffectivePhotonFlux", value=0.0)
        devsim.set_parameter(name="alpha", value=0.0)
        devsim.node_model(device=device, region=region, name="OpticalGeneration",
                          equation="alpha * EffectivePhotonFlux * exp(alpha * y)")

        # **DEFINITIVE FIX**: Create a single model for the net recombination/generation term.
        devsim.node_model(device=device, region=region, name="NetGenerationRecombination",
                          equation="-SRHRecombination + OpticalGeneration")

        # --- 3. Define Governing Equations (MUST BE LAST) ---

        # a) Poisson's Equation
        devsim.edge_model(device=device, region=region, name="ElectricField",
                          equation="(Potential@n0 - Potential@n1)*EdgeInverseLength")
        devsim.edge_model(device=device, region=region, name="PotentialEdgeFlux",
                          equation="Permittivity*ElectricField")

        devsim.equation(device=device, region=region, name="PotentialEquation",
                        variable_name="Potential",
                        edge_model="PotentialEdgeFlux",
                        node_model="NegativeSpaceCharge",
                        variable_update="log_damp")

        # b) Drift-Diffusion and Continuity Equations
        vdiff_eq = "(Potential@n0 - Potential@n1)/Vt"
        devsim.edge_model(device=device, region=region, name="vdiff", equation=vdiff_eq)

        devsim.edge_model(device=device, region=region, name="ElectronCurrent",
                          equation="mu_n*EdgeCouple*(Electrons@n0*B(vdiff) - Electrons@n1*B(-vdiff))")
        devsim.edge_model(device=device, region=region, name="HoleCurrent",
                          equation="-mu_p*EdgeCouple*(Holes@n0*B(-vdiff) - Holes@n1*B(vdiff))")

        # **FIXED**: Reference the single, combined model name instead of an expression.
        devsim.equation(device=device, region=region, name="ElectronContinuityEquation",
                        variable_name="Electrons", time_node_model="Electrons",
                        edge_model="ElectronCurrent",
                        node_model="NetGenerationRecombination")
        devsim.equation(device=device, region=region, name="HoleContinuityEquation",
                        variable_name="Holes", time_node_model="Holes",
                        edge_model="HoleCurrent",
                        node_model="NetGenerationRecombination")

    print("  All physics models and equations defined.")


def setup_boundary_conditions(device, material_params):
    """Defines ohmic contacts for anode and cathode."""
    print("\n--- Setting Up Boundary Conditions ---")
    for contact in ["anode", "cathode"]:
        devsim.contact_equation(device=device, contact=contact, name="PotentialEquation",
                                variable_name="Potential", node_model=f"Potential - {contact}_bias")
        devsim.contact_equation(device=device, contact=contact, name="ElectronContinuityEquation",
                                variable_name="Electrons", node_model="Electrons - n_eq")
        devsim.contact_equation(device=device, contact=contact, name="HoleContinuityEquation",
                                variable_name="Holes", node_model="Holes - p_eq")
    print("  Ohmic contacts set for anode and cathode.")


def debug_doping_profile(device):
    """Prints key values of the doping profile for verification."""
    p_doping = devsim.get_node_model_values(device=device, region="p_region", name="NetDoping")
    n_doping = devsim.get_node_model_values(device=device, region="n_region", name="NetDoping")
    print(f"  P-Region Doping Range: [{np.min(p_doping):.2e}, {np.max(p_doping):.2e}] cm⁻³")
    print(f"  N-Region Doping Range: [{np.min(n_doping):.2e}, {np.max(n_doping):.2e}] cm⁻³")

