#physics_setup.py
# !/usr/bin/env python3
"""
physics_setup.py - Silicon Photodiode Physics Models

Clean, robust implementation of semiconductor physics for photodiode simulation.
All equations from peer-reviewed literature with proper references.
Prioritizes reliability and correctness over advanced features.
"""

import devsim
import numpy as np


def bernoulli_deriv_stable(v):
    """
    Numerically stable Python implementation of the Bernoulli function's derivative.
    Handles the v=0 case where the standard formula would divide by zero.
    """
    # Create an output array of the same shape as the input
    out = np.zeros_like(v)

    # Identify where v is non-zero to avoid division by zero
    non_zero_mask = np.abs(v) > 1e-12
    v_nz = v[non_zero_mask]

    # Calculate the derivative for the non-zero values using the full formula
    exp_v = np.exp(v_nz)
    out[non_zero_mask] = (exp_v - 1.0 - v_nz * exp_v) / ((exp_v - 1.0) ** 2)

    # For values of v close to zero, use the known mathematical limit of -0.5
    out[~non_zero_mask] = -0.5

    return out


def setup_physics_and_materials(device, material_params, global_params):
    """
    Main function to set up all physics models in the correct order.
    """
    print("\n--- Setting up physics and materials ---")

    temperature_K = global_params["temperature_K"]

    # Step 1: Set fundamental parameters
    set_global_parameters(temperature_K)

    devsim.register_function(name="bernoulli_deriv_stable", function=bernoulli_deriv_stable)


    # Step 2: Process each region
    for region in ["p_region", "n_region"]:
        print(f"\nConfiguring {region}:")

        # Material parameters
        set_material_parameters(device, region, material_params)

        # Solution variables
        create_solution_variables(device, region)

        # Doping profile
        define_doping_profile(device, region, material_params)

        # Basic semiconductor models
        define_intrinsic_concentration(device, region, temperature_K)

        # --- CORRECTED ORDER ---
        # Define E-field FIRST, as mobility depends on it.
        define_electric_field(device, region)

        # Now define mobility, which can safely use EParallel.
        define_mobility_models(device, region, material_params)
        # --- END CORRECTION ---

        define_lifetime_models(device, region, material_params)

        # Transport models
        define_current_densities(device, region)

        # Generation-Recombination
        define_srh_recombination(device, region)
        define_auger_recombination(device, region, material_params)
        define_optical_generation(device, region)
        define_net_recombination(device, region)

        # Equilibrium models
        define_equilibrium_carriers(device, region)

        print(f"  ✓ {region} physics configured")

    print("\n--- Physics setup complete ---")

def set_global_parameters(temperature_K):
    """
    Set global simulation parameters.
    """
    # Physical constants
    devsim.set_parameter(name="q", value=1.602176634e-19)  # Elementary charge [C]
    devsim.set_parameter(name="k_B", value=1.380649e-23)  # Boltzmann constant [J/K]
    devsim.set_parameter(name="k_B_eV", value=8.617333e-5)  # Boltzmann constant [eV/K]

    # Temperature
    devsim.set_parameter(name="T", value=temperature_K)

    # Thermal voltage kT/q
    V_t = 8.617333e-5 * temperature_K  # in eV/eV = V
    devsim.set_parameter(name="ThermalVoltage", value=V_t)

    # Optical parameters (will be set during simulation)
    devsim.set_parameter(name="alpha", value=0.0)  # Absorption coefficient [cm^-1]
    devsim.set_parameter(name="EffectivePhotonFlux", value=0.0)  # [photons/cm^2/s]


def set_material_parameters(device, region, material_params):
    """
    Set material-specific parameters for silicon.
    """
    if region == "p_region":
        # The p_region must be defined as silicon with p-type behavior
        devsim.set_parameter(device=device, name="material", value="p-silicon")
    else:  # n_region
        # The n_region is correctly defined as n-type
        devsim.set_parameter(device=device, name="material", value="n-silicon")

    # Permittivity
    devsim.set_parameter(device=device, region=region,
                         name="Permittivity", value=material_params["permittivity"])

    # Charge
    devsim.set_parameter(device=device, region=region,
                         name="ElectronCharge", value=material_params["electron_charge"])

    # Density of states
    devsim.set_parameter(name="Nc_300K", value=material_params["Nc_300K"])
    devsim.set_parameter(name="Nv_300K", value=material_params["Nv_300K"])

    # Bandgap at 300K
    devsim.set_parameter(name="Eg_300K", value=material_params["bandgap"])


def create_solution_variables(device, region):
    """
    Create the three solution variables: Potential, Electrons, Holes.
    """
    devsim.node_solution(device=device, region=region, name="Potential")
    devsim.node_solution(device=device, region=region, name="Electrons")
    devsim.node_solution(device=device, region=region, name="Holes")


# In physics_setup.py, replace your existing function with this one:

def define_doping_profile(device, region, material_params):
    """
    Define doping profile for the photodiode with a robust, unit-consistent equation.
    """
    n_bulk = material_params["n_bulk"]

    if region == "p_region":
        # --- ROBUST IMPLEMENTATION ---

        # 1. Get parameters and convert them to centimeters immediately in Python.
        p_peak_cm = material_params["peak_p_doping"]
        p_depth_cm = material_params["projected_range"] * 1e-4  # Convert μm to cm
        p_straggle_cm = material_params["doping_straggle"] * 1e-4 # Convert μm to cm

        # 2. Build the equation string using direct numerical values.
        #    We also convert the mesh coordinate 'y' (in μm) to 'y*1e-4' (in cm)
        #    so all units inside the equation are consistently in centimeters.
        gaussian_eq = f"{p_peak_cm} * exp(-0.5 * pow((abs(y*1e-4) - {p_depth_cm}) / {p_straggle_cm}, 2))"

        # 3. Create the node models.
        devsim.node_model(device=device, region=region, name="Acceptors", equation=gaussian_eq)
        devsim.node_model(device=device, region=region, name="Donors", equation=f"{n_bulk}")

    else:  # n_region
        devsim.node_model(device=device, region=region, name="Acceptors", equation="0.0")
        devsim.node_model(device=device, region=region, name="Donors", equation=f"{n_bulk}")

    # Define NetDoping and TotalDoping for the region
    devsim.node_model(device=device, region=region, name="NetDoping",
                      equation="Donors - Acceptors")
    devsim.node_model(device=device, region=region, name="TotalDoping",
                      equation="abs(Donors) + abs(Acceptors)")

def define_intrinsic_concentration(device, region, temperature_K):
    """
    Temperature-dependent intrinsic carrier concentration.
    Ref: Green, J. Appl. Phys. 67, 2944 (1990)
    """
    # Temperature-dependent bandgap (Varshni equation for Si)
    Eg_eq = "1.17 - (4.73e-4 * T * T) / (T + 636.0)"
    devsim.node_model(device=device, region=region, name="Bandgap", equation=Eg_eq)

    # Intrinsic concentration: ni² = Nc*Nv*(T/300)³*exp(-Eg/kT)
    ni_eq = "(Nc_300K * Nv_300K * pow(T/300.0, 3.0))^(0.5) * exp(-Bandgap / (2.0 * k_B_eV * T))"
    devsim.node_model(device=device, region=region, name="IntrinsicCarrierDensity", equation=ni_eq)
    devsim.node_model(device=device, region=region, name="n_i_squared",
                      equation="IntrinsicCarrierDensity^2")


def define_mobility_models(device, region, material_params):
    """
    Caughey-Thomas mobility model with field dependence.
    Ref: Caughey & Thomas, Proc. IEEE 55, 2192 (1967)
    """
    # Low-field mobility (doping-dependent)
    mu_max_n = material_params["mu_max_n"]
    mu_min_n = material_params["mu_min_n"]
    N_ref_n = material_params["N_ref_mob_n"]
    alpha_n = material_params["alpha_mob_n"]

    mu_max_p = material_params["mu_max_p"]
    mu_min_p = material_params["mu_min_p"]
    N_ref_p = material_params["N_ref_mob_p"]
    alpha_p = material_params["alpha_mob_p"]

    # Node models for low-field mobility
    mu_n_eq = f"{mu_min_n} + ({mu_max_n} - {mu_min_n}) / (1 + pow(TotalDoping / {N_ref_n}, {alpha_n}))"
    mu_p_eq = f"{mu_min_p} + ({mu_max_p} - {mu_min_p}) / (1 + pow(TotalDoping / {N_ref_p}, {alpha_p}))"

    devsim.node_model(device=device, region=region, name="LowFieldElectronMobility", equation=mu_n_eq)
    devsim.node_model(device=device, region=region, name="LowFieldHoleMobility", equation=mu_p_eq)

    # Edge models for field-dependent mobility
    devsim.edge_average_model(device=device, region=region,
                              node_model="LowFieldElectronMobility",
                              edge_model="LowFieldElectronMobility_edge")
    devsim.edge_average_model(device=device, region=region,
                              node_model="LowFieldHoleMobility",
                              edge_model="LowFieldHoleMobility_edge")

    # Velocity saturation
    v_sat_n = material_params["v_sat_n"]
    v_sat_p = material_params["v_sat_p"]
    beta_n = material_params["beta_n"]
    beta_p = material_params["beta_p"]

    # Edge models with field dependence
    mu_n_field = f"LowFieldElectronMobility_edge / pow(1 + pow(LowFieldElectronMobility_edge * EParallel / {v_sat_n}, {beta_n}), 1/{beta_n})"
    mu_p_field = f"LowFieldHoleMobility_edge / pow(1 + pow(LowFieldHoleMobility_edge * EParallel / {v_sat_p}, {beta_p}), 1/{beta_p})"

    devsim.edge_model(device=device, region=region, name="ElectronMobility", equation=mu_n_field)
    devsim.edge_model(device=device, region=region, name="HoleMobility", equation=mu_p_field)


def define_lifetime_models(device, region, material_params):
    """
    Klaassen model for doping-dependent SRH lifetimes.
    Ref: Klaassen, Solid-State Electronics 35, 125 (1992)
    """
    tau_max_n = material_params["tau_max_n"]
    N_ref_n = material_params["N_ref_n"]
    tau_max_p = material_params["tau_max_p"]
    N_ref_p = material_params["N_ref_p"]

    tau_n_eq = f"{tau_max_n} / (1 + TotalDoping / {N_ref_n})"
    tau_p_eq = f"{tau_max_p} / (1 + TotalDoping / {N_ref_p})"

    devsim.node_model(device=device, region=region, name="taun", equation=tau_n_eq)
    devsim.node_model(device=device, region=region, name="taup", equation=tau_p_eq)


def define_electric_field(device, region):
    """
    Define electric field and related edge models.
    """
    # Get potential on edges
    devsim.edge_from_node_model(device=device, region=region, node_model="Potential")

    # Electric field E = -grad(Potential)
    devsim.edge_model(device=device, region=region, name="ElectricField",
                      equation="(Potential@n0 - Potential@n1) * EdgeInverseLength")


    # Parallel component (magnitude) for mobility calculation
    devsim.edge_model(device=device, region=region, name="EParallel",
                      equation="abs(ElectricField)")

    # Displacement field D = ε*E
    devsim.edge_model(device=device, region=region, name="DField",
                      equation="Permittivity * ElectricField")

    # Derivatives for Newton solver
    devsim.edge_model(device=device, region=region, name="DField:Potential@n0",
                      equation="Permittivity * EdgeInverseLength")
    devsim.edge_model(device=device, region=region, name="DField:Potential@n1",
                      equation="-Permittivity * EdgeInverseLength")


def define_current_densities(device, region):
    """
    Define electron and hole current densities using Scharfetter-Gummel
    discretization with a simple, numerically stable denominator.
    """
    # Space charge for Poisson equation
    devsim.node_model(device=device, region=region, name="SpaceCharge",
                      equation="ElectronCharge * (Holes - Electrons + NetDoping)")

    # Bernoulli functions
    devsim.edge_model(device=device, region=region, name="vdiff",
                      equation="(Potential@n0 - Potential@n1) / ThermalVoltage")
    devsim.edge_model(device=device, region=region, name="Bernoulli_vdiff",
                      equation="B(vdiff)")
    devsim.edge_model(device=device, region=region, name="Bernoulli_neg_vdiff",
                      equation="B(-vdiff)")

    # --- FINAL, SIMPLE FIX: Add a small number to the denominator to prevent division by zero ---
    deriv_eq = "(exp(vdiff) - 1 - vdiff*exp(vdiff)) / ((exp(vdiff) - 1)^2 + 1e-30)"
    devsim.edge_model(device=device, region=region, name="Bernoulli_deriv_vdiff",
                      equation=deriv_eq)

    neg_vdiff = "(-vdiff)"
    neg_deriv_eq = f"(exp({neg_vdiff}) - 1 - ({neg_vdiff})*exp({neg_vdiff})) / ((exp({neg_vdiff}) - 1)^2 + 1e-30)"
    devsim.edge_model(device=device, region=region, name="Bernoulli_deriv_neg_vdiff",
                      equation=neg_deriv_eq)

    # Get carrier concentrations on edges
    devsim.edge_from_node_model(device=device, region=region, node_model="Electrons")
    devsim.edge_from_node_model(device=device, region=region, node_model="Holes")

    # Electron and Hole Current equations (unchanged)
    Jn_eq = "ElectronCharge * ElectronMobility * ThermalVoltage * EdgeInverseLength * (Electrons@n1 * Bernoulli_neg_vdiff - Electrons@n0 * Bernoulli_vdiff)"
    devsim.edge_model(device=device, region=region, name="ElectronCurrent", equation=Jn_eq)
    Jp_eq = "ElectronCharge * HoleMobility * ThermalVoltage * EdgeInverseLength * (Holes@n1 * Bernoulli_vdiff - Holes@n0 * Bernoulli_neg_vdiff)"
    devsim.edge_model(device=device, region=region, name="HoleCurrent", equation=Jp_eq)

    # Fully explicit, non-cyclic derivatives (unchanged)
    prefactor_n = "ElectronCharge * ElectronMobility * EdgeInverseLength"
    prefactor_p = "ElectronCharge * HoleMobility * EdgeInverseLength"

    devsim.edge_model(device=device, region=region, name="ElectronCurrent:Electrons@n0",
                      equation=f"-{prefactor_n} * ThermalVoltage * Bernoulli_vdiff")
    devsim.edge_model(device=device, region=region, name="ElectronCurrent:Electrons@n1",
                      equation=f"{prefactor_n} * ThermalVoltage * Bernoulli_neg_vdiff")
    devsim.edge_model(device=device, region=region, name="ElectronCurrent:Potential@n0",
                      equation=f"-{prefactor_n} * (Electrons@n1 * Bernoulli_deriv_neg_vdiff + Electrons@n0 * Bernoulli_deriv_vdiff)")
    devsim.edge_model(device=device, region=region, name="ElectronCurrent:Potential@n1",
                      equation=f"{prefactor_n} * (Electrons@n1 * Bernoulli_deriv_neg_vdiff + Electrons@n0 * Bernoulli_deriv_vdiff)")

    devsim.edge_model(device=device, region=region, name="HoleCurrent:Holes@n0",
                      equation=f"-{prefactor_p} * ThermalVoltage * Bernoulli_neg_vdiff")
    devsim.edge_model(device=device, region=region, name="HoleCurrent:Holes@n1",
                      equation=f"{prefactor_p} * ThermalVoltage * Bernoulli_vdiff")
    devsim.edge_model(device=device, region=region, name="HoleCurrent:Potential@n0",
                      equation=f"{prefactor_p} * (Holes@n1 * Bernoulli_deriv_vdiff + Holes@n0 * Bernoulli_deriv_neg_vdiff)")
    devsim.edge_model(device=device, region=region, name="HoleCurrent:Potential@n1",
                      equation=f"-{prefactor_p} * (Holes@n1 * Bernoulli_deriv_vdiff + Holes@n0 * Bernoulli_deriv_neg_vdiff)")

    devsim.edge_model(device=device, region=region, name="ElectronCurrent:Holes@n0", equation="0.0")
    devsim.edge_model(device=device, region=region, name="ElectronCurrent:Holes@n1", equation="0.0")
    devsim.edge_model(device=device, region=region, name="HoleCurrent:Electrons@n0", equation="0.0")
    devsim.edge_model(device=device, region=region, name="HoleCurrent:Electrons@n1", equation="0.0")



def define_srh_recombination(device, region):
    """
    Shockley-Read-Hall recombination.
    Ref: Shockley & Read, Phys. Rev. 87, 835 (1952)
    """
    # For midgap traps: n1 = p1 = ni
    devsim.node_model(device=device, region=region, name="n1_srh",
                      equation="IntrinsicCarrierDensity")
    devsim.node_model(device=device, region=region, name="p1_srh",
                      equation="IntrinsicCarrierDensity")

    # SRH recombination rate
    srh_eq = "(Electrons * Holes - n_i_squared) / (taup * (Electrons + n1_srh) + taun * (Holes + p1_srh))"
    devsim.node_model(device=device, region=region, name="USRH", equation=srh_eq)

    # Derivatives
    devsim.node_model(device=device, region=region, name="USRH:Electrons",
                      equation=f"diff({srh_eq}, Electrons)")
    devsim.node_model(device=device, region=region, name="USRH:Holes",
                      equation=f"diff({srh_eq}, Holes)")


def define_auger_recombination(device, region, material_params):
    """
    Auger recombination.
    Ref: Dziewior & Schmid, Appl. Phys. Lett. 31, 346 (1977)
    """
    C_n = material_params["C_n_auger"]
    C_p = material_params["C_p_auger"]

    devsim.set_parameter(device=device, region=region, name="C_n_auger", value=C_n)
    devsim.set_parameter(device=device, region=region, name="C_p_auger", value=C_p)

    auger_eq = "(C_n_auger * Electrons + C_p_auger * Holes) * (Electrons * Holes - n_i_squared)"
    devsim.node_model(device=device, region=region, name="UAuger", equation=auger_eq)

    # Derivatives
    devsim.node_model(device=device, region=region, name="UAuger:Electrons",
                      equation=f"diff({auger_eq}, Electrons)")
    devsim.node_model(device=device, region=region, name="UAuger:Holes",
                      equation=f"diff({auger_eq}, Holes)")


def define_optical_generation(device, region):
    """
    Optical generation using Beer-Lambert law.
    G = α * Φ * exp(-α * depth)
    """
    # y coordinates: 0 at surface, negative going into bulk
    # depth = |y| in cm
    optical_eq = "EffectivePhotonFlux * alpha * exp(-alpha * abs(y) * 1e-4)"

    devsim.node_model(device=device, region=region, name="OpticalGeneration",
                      equation=optical_eq)

    # No derivatives with respect to carriers
    devsim.node_model(device=device, region=region, name="OpticalGeneration:Electrons",
                      equation="0.0")
    devsim.node_model(device=device, region=region, name="OpticalGeneration:Holes",
                      equation="0.0")


def define_net_recombination(device, region):
    """
    Net recombination rate: R_net = R_SRH + R_Auger - G_optical
    """
    net_eq = "USRH + UAuger - OpticalGeneration"

    devsim.node_model(device=device, region=region, name="NetRecombination",
                      equation=net_eq)

    # Derivatives
    devsim.node_model(device=device, region=region, name="NetRecombination:Electrons",
                      equation="USRH:Electrons + UAuger:Electrons")
    devsim.node_model(device=device, region=region, name="NetRecombination:Holes",
                      equation="USRH:Holes + UAuger:Holes")

    # For continuity equations
    devsim.node_model(device=device, region=region, name="eCharge_x_NetRecomb",
                      equation="ElectronCharge * NetRecombination")
    devsim.node_model(device=device, region=region, name="Neg_eCharge_x_NetRecomb",
                      equation="-ElectronCharge * NetRecombination")


def define_equilibrium_carriers(device, region):
    """
    Equilibrium carrier concentrations for initial conditions.
    At equilibrium: n*p = ni² and n - p = N_D - N_A
    """
    # Equilibrium electron concentration
    n_eq = "0.5 * (NetDoping + (NetDoping^2 + 4*n_i_squared)^(0.5))"
    devsim.node_model(device=device, region=region, name="IntrinsicElectrons", equation=n_eq)

    # Equilibrium hole concentration
    p_eq = "0.5 * (-NetDoping + (NetDoping^2 + 4*n_i_squared)^(0.5))"
    devsim.node_model(device=device, region=region, name="IntrinsicHoles", equation=p_eq)


# ==============================================================================
# BOUNDARY CONDITIONS
# ==============================================================================

def setup_boundary_conditions(device, material_params):
    """
    Set up boundary conditions for contacts and interfaces.
    Simple ohmic contacts with no surface recombination for reliability.
    """
    print("\n--- Setting up boundary conditions ---")

    # Set surface recombination velocities (set to 0 for simplicity)
    devsim.set_parameter(device=device, name="Sn", value=0.0)
    devsim.set_parameter(device=device, name="Sp", value=0.0)

    # Contact boundary conditions
    for contact in ["anode", "cathode"]:
        # Bias parameter
        devsim.set_parameter(device=device, name=f"{contact}_bias", value=0.0)

        # Potential BC: ψ = ψ_applied
        devsim.contact_node_model(device=device, contact=contact,
                                  name=f"{contact}_potential_bc",
                                  equation=f"Potential - {contact}_bias")
        devsim.contact_node_model(device=device, contact=contact,
                                  name=f"{contact}_potential_bc:Potential",
                                  equation="1.0")

        # Carrier BCs: equilibrium concentrations
        devsim.contact_node_model(device=device, contact=contact,
                                  name=f"{contact}_electrons_bc",
                                  equation="Electrons - IntrinsicElectrons")
        devsim.contact_node_model(device=device, contact=contact,
                                  name=f"{contact}_electrons_bc:Electrons",
                                  equation="1.0")

        devsim.contact_node_model(device=device, contact=contact,
                                  name=f"{contact}_holes_bc",
                                  equation="Holes - IntrinsicHoles")
        devsim.contact_node_model(device=device, contact=contact,
                                  name=f"{contact}_holes_bc:Holes",
                                  equation="1.0")

    # Interface continuity conditions at p-n junction
    for variable in ["Potential", "Electrons", "Holes"]:
        devsim.interface_model(device=device, interface="pn_junction",
                               name=f"{variable}_continuity",
                               equation=f"{variable}@r0 - {variable}@r1")
        devsim.interface_model(device=device, interface="pn_junction",
                               name=f"{variable}_continuity:{variable}@r0",
                               equation="1.0")
        devsim.interface_model(device=device, interface="pn_junction",
                               name=f"{variable}_continuity:{variable}@r1",
                               equation="-1.0")

    print("  ✓ Boundary conditions configured")


# ==============================================================================
# EQUATION ASSEMBLY
# ==============================================================================

def setup_carrier_transport_equations(device):
    """
    Assemble the full coupled system of equations: Poisson, Electron Continuity,
    and Hole Continuity, including all boundary conditions with corrected naming.
    """
    print("\n--- Adding carrier transport equations ---")

    equations_to_define = [
        {
            "variable": "Potential", "equation_name": "PotentialEquation",
            "node_model": "SpaceCharge", "edge_model": "DField",
            "update": "default"
        },
        {
            "variable": "Electrons", "equation_name": "ElectronContinuityEquation",
            "node_model": "eCharge_x_NetRecomb", "edge_model": "ElectronCurrent",
            "update": "log_damp"
        },
        {
            "variable": "Holes", "equation_name": "HoleContinuityEquation",
            "node_model": "Neg_eCharge_x_NetRecomb",  # <-- THIS TYPO IS NOW FIXED
            "edge_model": "HoleCurrent",
            "update": "log_damp"
        }
    ]

    # 1. Define all three equations in the bulk regions
    for region in ["p_region", "n_region"]:
        for eq in equations_to_define:
            devsim.equation(
                device=device, region=region, name=eq["equation_name"],
                variable_name=eq["variable"], node_model=eq["node_model"],
                edge_model=eq["edge_model"], variable_update=eq["update"]
            )

    # 2. Define interface continuity using the CORRECT equation names
    for eq in equations_to_define:
        devsim.interface_equation(
            device=device, interface="pn_junction",
            name=eq["equation_name"],
            interface_model=f"{eq['variable']}_continuity", type="continuous"
        )

    # 3. Define contact boundary conditions for all three equations
    for contact in ["anode", "cathode"]:
        devsim.contact_equation(
            device=device, contact=contact, name="PotentialEquation",
            node_model=f"{contact}_potential_bc", edge_charge_model="DField"
        )
        devsim.contact_equation(
            device=device, contact=contact, name="ElectronContinuityEquation",
            node_model=f"{contact}_electrons_bc", edge_current_model="ElectronCurrent"
        )
        devsim.contact_equation(
            device=device, contact=contact, name="HoleContinuityEquation",
            node_model=f"{contact}_holes_bc", edge_current_model="HoleCurrent"
        )

    print("  ✓ Transport equations added")


# ==============================================================================
# OPTICAL PARAMETER HELPERS
# ==============================================================================

def set_optical_parameters(wavelength_nm, photon_flux, material_params):
    """
    Set optical generation parameters for a specific wavelength.

    Parameters:
    -----------
    wavelength_nm : float
        Wavelength in nanometers
    photon_flux : float
        Incident photon flux in photons/cm²/s
    material_params : dict
        Material parameters dictionary
    """
    # Get optical constants
    alpha = get_alpha_for_wavelength(wavelength_nm, material_params)
    reflectivity = get_reflectivity(wavelength_nm, material_params)

    # Effective flux after reflection
    effective_flux = photon_flux * (1.0 - reflectivity)

    # Set global parameters
    devsim.set_parameter(name="alpha", value=alpha)
    devsim.set_parameter(name="EffectivePhotonFlux", value=effective_flux)

    return alpha, reflectivity, effective_flux


def get_alpha_for_wavelength(wavelength_nm, material_params):
    """
    Calculate absorption coefficient from optical data.
    α = 4πk/λ where k is extinction coefficient
    """
    optical_data = material_params["optical_data"]

    # Interpolate extinction coefficient
    k_e = np.interp(wavelength_nm,
                    optical_data["wavelengths"],
                    optical_data["k_values"])

    # Calculate absorption coefficient [cm⁻¹]
    alpha_cm = (4 * np.pi * k_e) / (wavelength_nm * 1e-7)

    return alpha_cm


def get_reflectivity(wavelength_nm, material_params):
    """
    Calculate Fresnel reflectivity at air-silicon interface.
    R = |r|² where r = (n₁ - n₂)/(n₁ + n₂)
    """
    optical_data = material_params["optical_data"]

    # Interpolate refractive indices
    n_si = np.interp(wavelength_nm,
                     optical_data["wavelengths"],
                     optical_data["n_values"])
    k_si = np.interp(wavelength_nm,
                     optical_data["wavelengths"],
                     optical_data["k_values"])

    # Complex refractive index
    n_air = 1.0
    N_si = complex(n_si, -k_si)

    # Fresnel reflection coefficient
    r = (n_air - N_si) / (n_air + N_si)
    reflectivity = abs(r) ** 2

    return reflectivity

def debug_doping_profile(device_name):
    """
    Provides a comprehensive and robust debug output for the doping profile,
    with improved logic for junction finding that is insensitive to numerical noise.
    """
    print("\n" + "-"*40)
    print("      Doping Profile Analysis")
    print("-" * 40)

    try:
        # --- Analyze p-region (Implanted region) ---
        print("\nAnalyzing p_region:")
        y_raw = np.array(devsim.get_node_model_values(device=device_name, region="p_region", name="y"))
        acceptors_raw = np.array(devsim.get_node_model_values(device=device_name, region="p_region", name="Acceptors"))
        net_doping_raw = np.array(devsim.get_node_model_values(device=device_name, region="p_region", name="NetDoping"))

        # Sort all data by depth (y-coordinate) to ensure correct processing
        sort_indices = np.argsort(y_raw)
        y_coords = y_raw[sort_indices]
        acceptors = acceptors_raw[sort_indices]
        net_doping = net_doping_raw[sort_indices]

        # Find key locations
        surface_idx = np.argmin(np.abs(y_coords))
        peak_idx = np.argmax(acceptors)

        print(f"  - Surface (y~0 μm) Doping: N_A = {acceptors[surface_idx]:.2e} cm⁻³")
        print(f"  - Peak P-type Doping:    Peak N_A = {acceptors[peak_idx]:.2e} cm⁻³ at depth y = {abs(y_coords[peak_idx]):.3f} μm")

        # --- Junction Finding Logic (robust & direction-aware) ---
        # We determine the junction by scanning from the surface (|y| minimum) into the bulk.
        # This avoids false "entire region n-type" warnings when the deep tail is lightly n-type.
        order_from_surface = np.argsort(np.abs(y_coords))
        y_surf_to_deep = y_coords[order_from_surface]
        net_surf_to_deep = net_doping[order_from_surface]

        # Determine the sign at the surface (just inside the p-region)
        sign_surface = np.sign(net_surf_to_deep[0])

        # Find the first index where the sign differs from the surface sign
        sign_change_indices = np.where(np.sign(net_surf_to_deep) != sign_surface)[0]

        if sign_change_indices.size > 0:
            idx2 = sign_change_indices[0]
            idx1 = idx2 - 1

            # Linear interpolation to find y where NetDoping crosses zero
            y1, y2 = y_surf_to_deep[idx1], y_surf_to_deep[idx2]
            n1, n2 = net_surf_to_deep[idx1], net_surf_to_deep[idx2]
            junction_depth_um = abs(y1 - n1 * (y2 - y1) / (n2 - n1))
            print(f"  - Metallurgical Junction: ✅ Junction found at depth xj = {junction_depth_um:.3f} μm")
        else:
            # No sign change from the surface to the bottom of the p-region
            if sign_surface > 0:
                print("    ⚠️ WARNING: Entire p-region appears n-type starting at the surface. Check acceptor profile or donor background.")
            elif sign_surface < 0:
                print("    ⚠️ WARNING: The entire p-region is p-type. Junction is deeper than the defined p-region mesh.")
            else:
                print("    ℹ️ Note: NetDoping is ~0 near the surface; cannot determine junction reliably here.")

    except Exception as e:
        print(f"  ❌ ERROR analyzing p-region: {e}")

    # --- Analyze n-region (Substrate) ---
    try:
        print("Analyzing n_region:")
        n_donors = np.array(devsim.get_node_model_values(device=device_name, region="n_region", name="Donors"))
        print(f"  - Bulk N-type Doping:   Uniform N_D ≈ {np.mean(n_donors):.2e} cm⁻³")
    except Exception as e:
        print(f"  ❌ ERROR analyzing n_region: {e}")

    print("-" * 40 + "\n")

