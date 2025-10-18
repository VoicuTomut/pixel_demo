# Copyright 2013 DEVSIM LLC
#
# SPDX-License-Identifier: Apache-2.0

"""
Semiconductor Device Physics Models
====================================
This module implements drift-diffusion transport equations for semiconductor
device simulation, including:
- Poisson's equation for electrostatics
- Electron and hole continuity equations
- Shockley-Read-Hall (SRH) recombination
- Contact boundary conditions
"""

from .simple_dd import CreateBernoulli, CreateElectronCurrent, CreateHoleCurrent
from .model_create import (
    CreateSolution,
    CreateNodeModel,
    CreateNodeModelDerivative,
    CreateContactNodeModel,
    CreateContactNodeModelDerivative,
    CreateEdgeModel,
    CreateEdgeModelDerivatives,
    CreateContinuousInterfaceModel,
    InEdgeModelList,
    InNodeModelList,
)

from devsim import (
    contact_equation,
    equation,
    get_contact_current,
    get_parameter,
    interface_equation,
    set_parameter,
)

contactcharge_edge = "contactcharge_edge"
ece_name = "ElectronContinuityEquation"
hce_name = "HoleContinuityEquation"

# Equilibrium carrier concentration models
# For n-type (N_D > 0): n ≈ N_D + sqrt(N_D^2 + 4n_i^2)/2
# For p-type (N_D < 0): p ≈ -N_D + sqrt(N_D^2 + 4n_i^2)/2
# Small constant (1e-10) added for numerical stability
celec_model = "(1e-10 + 0.5*abs(NetDoping+(NetDoping^2 + 4 * n_i^2)^(0.5)))"
chole_model = "(1e-10 + 0.5*abs(-NetDoping+(NetDoping^2 + 4 * n_i^2)^(0.5)))"

# ============================================================================
# Physical Constants
# ============================================================================
q = 1.6e-19  # Elementary charge [Coulombs]
k = 1.3806503e-23  # Boltzmann constant [J/K]
eps_0 = 8.85e-14  # Vacuum permittivity [F/cm²]
eps_si = 11.1  # Relative permittivity of silicon (dimensionless)
eps_ox = 3.9  # Relative permittivity of SiO₂ (dimensionless)
n_i = 1.0e10  # Intrinsic carrier concentration [cm⁻³] at 300K
mu_n = 400  # Electron mobility [cm²/(V·s)]
mu_p = 200  # Hole mobility [cm²/(V·s)]


def GetContactBiasName(contact):
    """Returns the parameter name for contact bias voltage."""
    return "{0}_bias".format(contact)


def GetContactNodeModelName(contact):
    """Returns the node model name for contact boundary condition."""
    return "{0}nodemodel".format(contact)


def PrintCurrents(device, contact):
    """
    Print contact currents for diagnostics.

    Total current I = I_n + I_p where:
    - I_n: electron current (negative charge moving creates positive current)
    - I_p: hole current (positive charge moving creates positive current)
    """
    electron_current = get_contact_current(
        device=device, contact=contact, equation=ece_name
    )
    hole_current = get_contact_current(
        device=device, contact=contact, equation=hce_name
    )
    total_current = electron_current + hole_current
    voltage = get_parameter(device=device, name=GetContactBiasName(contact))
    print(
        "{0}\t{1}\t{2}\t{3}\t{4}".format(
            contact, voltage, electron_current, hole_current, total_current
        )
    )


def CreateOxideContact(device, region, contact):
    """
    Create boundary condition for metal-oxide contact.

    Physics:
    -------
    At a metal contact on oxide, the potential is fixed to the applied bias:
        Φ = V_contact

    The normal electric field at the contact determines the surface charge:
        σ = ε·E_n = -ε·∇Φ·n̂
    """
    contact_bias_name = GetContactBiasName(contact)
    contact_model_name = GetContactNodeModelName(contact)

    # Boundary condition: Potential equals applied bias
    eq = "Potential - {0}".format(contact_bias_name)
    CreateContactNodeModel(device, contact, contact_model_name, eq)
    CreateContactNodeModelDerivative(
        device, contact, contact_model_name, eq, "Potential"
    )

    # Edge model for displacement field: D = ε·E = ε·∇Φ
    if not InEdgeModelList(device, region, contactcharge_edge):
        CreateEdgeModel(
            device, region, contactcharge_edge, "Permittivity*ElectricField"
        )
        CreateEdgeModelDerivatives(
            device,
            region,
            contactcharge_edge,
            "Permittivity*ElectricField",
            "Potential",
        )

    contact_equation(
        device=device,
        contact=contact,
        name="PotentialEquation",
        node_model=contact_model_name,
        edge_charge_model=contactcharge_edge,
    )


def SetOxideParameters(device, region, T):
    """
    Set physical parameters for silicon dioxide (SiO₂) region.

    Parameters:
    ----------
    Permittivity: ε = ε_r·ε₀ = 3.9 × 8.85×10⁻¹⁴ F/cm²
    """
    set_parameter(
        device=device, region=region, name="Permittivity", value=eps_ox * eps_0
    )
    set_parameter(device=device, region=region, name="ElectronCharge", value=q)


def SetSiliconParameters(device, region, T):
    """
    Set physical parameters for silicon region.

    Key Parameters:
    --------------
    - Permittivity: ε = ε_r·ε₀ = 11.1 × 8.85×10⁻¹⁴ F/cm²
    - Thermal voltage: V_t = kT/q ≈ 26 mV at 300K
    - Intrinsic concentration: n_i ≈ 10¹⁰ cm⁻³ at 300K (n·p = n_i²)
    - Mobilities: μ_n ≈ 400 cm²/(V·s), μ_p ≈ 200 cm²/(V·s)
    - SRH lifetimes: τ_n, τ_p ≈ 10 μs (typical values)
    """
    set_parameter(
        device=device, region=region, name="Permittivity", value=eps_si * eps_0
    )
    set_parameter(device=device, region=region, name="ElectronCharge", value=q)
    set_parameter(device=device, region=region, name="n_i", value=n_i)
    set_parameter(device=device, region=region, name="T", value=T)
    set_parameter(device=device, region=region, name="kT", value=k * T)
    set_parameter(device=device, region=region, name="V_t", value=k * T / q)
    set_parameter(device=device, region=region, name="mu_n", value=mu_n)
    set_parameter(device=device, region=region, name="mu_p", value=mu_p)

    # SRH recombination parameters
    # n1, p1: reference concentrations (typically = n_i)
    # τ_n, τ_p: carrier lifetimes
    set_parameter(device=device, region=region, name="n1", value=n_i)
    set_parameter(device=device, region=region, name="p1", value=n_i)
    set_parameter(device=device, region=region, name="taun", value=1e-5)
    set_parameter(device=device, region=region, name="taup", value=1e-5)


def CreateSiliconPotentialOnly(device, region):
    """
    Create Poisson equation for electrostatics only (no carrier transport).

    Poisson's Equation:
    ------------------
    ∇·(ε∇Φ) = -ρ = -q(p - n + N_D - N_A)

    where:
    - Φ: electrostatic potential [V]
    - ε: permittivity [F/cm²]
    - ρ: charge density [C/cm³]
    - n, p: electron and hole concentrations [cm⁻³]
    - N_D - N_A: net doping (donor - acceptor) [cm⁻³]

    At equilibrium (no transport):
    - n = n_i·exp(Φ/V_t): Boltzmann distribution for electrons
    - p = n_i·exp(-Φ/V_t) = n_i²/n: mass action law
    - ρ = q(p - n + N_D - N_A)
    """
    if not InNodeModelList(device, region, "Potential"):
        print("Creating Node Solution Potential")
        CreateSolution(device, region, "Potential")

    # Intrinsic carrier concentrations at local potential
    # n = n_i·exp(qΦ/kT) = n_i·exp(Φ/V_t)
    elec_i = "n_i*exp(Potential/V_t)"

    # p = n_i²/n (mass action law: n·p = n_i²)
    hole_i = "n_i^2/IntrinsicElectrons"

    # Net charge density: ρ = q(p - n + N_D)
    # Using Kahan summation for numerical accuracy
    charge_i = "kahan3(IntrinsicHoles, -IntrinsicElectrons, NetDoping)"

    # Charge density in Coulombs: ρ[C/cm³] = -q·charge_density[cm⁻³]
    pcharge_i = "-ElectronCharge * IntrinsicCharge"

    # Create node models for charge calculation
    for i in (
            ("IntrinsicElectrons", elec_i),
            ("IntrinsicHoles", hole_i),
            ("IntrinsicCharge", charge_i),
            ("PotentialIntrinsicCharge", pcharge_i),
    ):
        n = i[0]
        e = i[1]
        CreateNodeModel(device, region, n, e)
        CreateNodeModelDerivative(device, region, n, e, "Potential")

    # Electric field on edges: E = -∇Φ ≈ -(Φ_n1 - Φ_n0)/Δx
    # Displacement flux: D = ε·E [C/cm²]
    for i in (
            ("ElectricField", "(Potential@n0-Potential@n1)*EdgeInverseLength"),
            ("PotentialEdgeFlux", "Permittivity * ElectricField"),
    ):
        n = i[0]
        e = i[1]
        CreateEdgeModel(device, region, n, e)
        CreateEdgeModelDerivatives(device, region, n, e, "Potential")

    # Assemble Poisson equation: ∇·(ε∇Φ) = -ρ
    equation(
        device=device,
        region=region,
        name="PotentialEquation",
        variable_name="Potential",
        node_model="PotentialIntrinsicCharge",
        edge_model="PotentialEdgeFlux",
        variable_update="log_damp",
    )


def CreateSiliconPotentialOnlyContact(device, region, contact, is_circuit=False):
    """
    Create contact boundary condition for Poisson equation.

    Contact Boundary Condition:
    --------------------------
    At thermal equilibrium, the potential at a contact is determined by
    the work function and doping:

    For n-type: Φ_c = V_bias - V_t·ln(n_eq/n_i)
    For p-type: Φ_c = V_bias + V_t·ln(p_eq/n_i)

    where V_t = kT/q is the thermal voltage and n_eq, p_eq are the
    equilibrium carrier concentrations.

    The built-in potential accounts for the Fermi level position:
    V_bi = ±V_t·ln(N_doping/n_i)
    """
    # Edge charge model: σ = ε·E = ε·∇Φ
    if not InEdgeModelList(device, region, "contactcharge_edge"):
        CreateEdgeModel(
            device, region, "contactcharge_edge", "Permittivity*ElectricField"
        )
        CreateEdgeModelDerivatives(
            device,
            region,
            "contactcharge_edge",
            "Permittivity*ElectricField",
            "Potential",
        )

    # Contact potential with built-in voltage
    # Φ = V_bias ∓ V_t·ln(n_eq/n_i) for n-type/p-type
    contact_model = "Potential -{0} + ifelse(NetDoping > 0, \
    -V_t*log({1}/n_i), \
    V_t*log({2}/n_i))".format(GetContactBiasName(contact), celec_model, chole_model)

    contact_model_name = GetContactNodeModelName(contact)
    CreateContactNodeModel(device, contact, contact_model_name, contact_model)

    # Derivatives for Newton solver
    CreateContactNodeModel(
        device, contact, "{0}:{1}".format(contact_model_name, "Potential"), "1"
    )
    if is_circuit:
        CreateContactNodeModel(
            device,
            contact,
            "{0}:{1}".format(contact_model_name, GetContactBiasName(contact)),
            "-1",
        )

    if is_circuit:
        contact_equation(
            device=device,
            contact=contact,
            name="PotentialEquation",
            node_model=contact_model_name,
            edge_model="",
            node_charge_model="",
            edge_charge_model="contactcharge_edge",
            node_current_model="",
            edge_current_model="",
            circuit_node=GetContactBiasName(contact),
        )
    else:
        contact_equation(
            device=device,
            contact=contact,
            name="PotentialEquation",
            node_model=contact_model_name,
            edge_model="",
            node_charge_model="",
            edge_charge_model="contactcharge_edge",
            node_current_model="",
            edge_current_model="",
        )


def CreateSRH(device, region):
    """
    Create Shockley-Read-Hall recombination model.

    SRH Recombination:
    -----------------
    The net recombination rate through trap-assisted processes is:

    U_SRH = (n·p - n_i²) / [τ_p·(n + n₁) + τ_n·(p + p₁)]

    where:
    - n, p: electron and hole concentrations [cm⁻³]
    - n_i: intrinsic concentration [cm⁻³]
    - τ_n, τ_p: electron and hole lifetimes [s]
    - n₁, p₁: trap reference concentrations ≈ n_i [cm⁻³]

    Generation/Recombination rates:
    - R > 0 when n·p > n_i² (recombination)
    - R < 0 when n·p < n_i² (generation)

    Impact on continuity equations:
    - G_n = -q·U_SRH (electrons gained from recombination)
    - G_p = +q·U_SRH (holes gained from recombination)
    """
    USRH = "(Electrons*Holes - n_i^2)/(taup*(Electrons + n1) + taun*(Holes + p1))"
    Gn = "-ElectronCharge * USRH"  # Generation for electrons (negative charge)
    Gp = "+ElectronCharge * USRH"  # Generation for holes (positive charge)

    CreateNodeModel(device, region, "USRH", USRH)
    CreateNodeModel(device, region, "ElectronGeneration", Gn)
    CreateNodeModel(device, region, "HoleGeneration", Gp)

    # Derivatives needed for Newton solver
    for i in ("Electrons", "Holes"):
        CreateNodeModelDerivative(device, region, "USRH", USRH, i)
        CreateNodeModelDerivative(device, region, "ElectronGeneration", Gn, i)
        CreateNodeModelDerivative(device, region, "HoleGeneration", Gp, i)


def CreateECE(device, region, mu_n):
    """
    Create Electron Continuity Equation.

    Electron Continuity Equation:
    ----------------------------
    ∂n/∂t = (1/q)∇·J_n + G_n

    where:
    - n: electron concentration [cm⁻³]
    - J_n: electron current density [A/cm²]
    - G_n: net generation rate [cm⁻³·s⁻¹]

    Drift-Diffusion Current:
    J_n = q·μ_n·n·E + q·D_n·∇n
        = q·μ_n·n·∇Φ + q·D_n·∇n  (E = -∇Φ)

    where μ_n is mobility and D_n = μ_n·V_t (Einstein relation).

    Charge conservation: ∂ρ_n/∂t + ∇·J_n = 0
    where ρ_n = -q·n
    """
    CreateElectronCurrent(device, region, mu_n)

    # Electron charge density: ρ_n = -q·n [C/cm³]
    NCharge = "-ElectronCharge * Electrons"
    CreateNodeModel(device, region, "NCharge", NCharge)
    CreateNodeModelDerivative(device, region, "NCharge", NCharge, "Electrons")

    # Time-dependent continuity equation
    equation(
        device=device,
        region=region,
        name="ElectronContinuityEquation",
        variable_name="Electrons",
        time_node_model="NCharge",
        edge_model="ElectronCurrent",
        variable_update="positive",
        node_model="ElectronGeneration",
    )


def CreateHCE(device, region, mu_p):
    """
    Create Hole Continuity Equation.

    Hole Continuity Equation:
    ------------------------
    ∂p/∂t = -(1/q)∇·J_p + G_p

    where:
    - p: hole concentration [cm⁻³]
    - J_p: hole current density [A/cm²]
    - G_p: net generation rate [cm⁻³·s⁻¹]

    Drift-Diffusion Current:
    J_p = q·μ_p·p·E - q·D_p·∇p
        = -q·μ_p·p·∇Φ - q·D_p·∇p

    where μ_p is mobility and D_p = μ_p·V_t (Einstein relation).

    Note the sign difference from electrons due to positive charge.

    Charge conservation: ∂ρ_p/∂t + ∇·J_p = 0
    where ρ_p = +q·p
    """
    CreateHoleCurrent(device, region, mu_p)

    # Hole charge density: ρ_p = +q·p [C/cm³]
    PCharge = "ElectronCharge * Holes"
    CreateNodeModel(device, region, "PCharge", PCharge)
    CreateNodeModelDerivative(device, region, "PCharge", PCharge, "Holes")

    # Time-dependent continuity equation
    equation(
        device=device,
        region=region,
        name="HoleContinuityEquation",
        variable_name="Holes",
        time_node_model="PCharge",
        edge_model="HoleCurrent",
        variable_update="positive",
        node_model="HoleGeneration",
    )


def CreatePE(device, region):
    """
    Create Poisson Equation for drift-diffusion simulation.

    Poisson's Equation:
    ------------------
    ∇·(ε∇Φ) = -ρ = -q(p - n + N_D - N_A)

    This couples the potential to the carrier concentrations, which are
    now solution variables (not equilibrium distributions).

    The space charge density includes:
    - Mobile electrons: -q·n
    - Mobile holes: +q·p
    - Ionized donors: +q·N_D⁺ ≈ +q·N_D
    - Ionized acceptors: -q·N_A⁻ ≈ -q·N_A

    Using Kahan summation for numerical accuracy in charge calculation.
    """
    # Total charge density: ρ = q(p - n + N_D - N_A)
    pne = "-ElectronCharge*kahan3(Holes, -Electrons, NetDoping)"
    CreateNodeModel(device, region, "PotentialNodeCharge", pne)
    CreateNodeModelDerivative(device, region, "PotentialNodeCharge", pne, "Electrons")
    CreateNodeModelDerivative(device, region, "PotentialNodeCharge", pne, "Holes")

    # Assemble Poisson equation with carrier coupling
    equation(
        device=device,
        region=region,
        name="PotentialEquation",
        variable_name="Potential",
        node_model="PotentialNodeCharge",
        edge_model="PotentialEdgeFlux",
        time_node_model="",
        variable_update="log_damp",
    )


def CreateSiliconDriftDiffusion(device, region, mu_n="mu_n", mu_p="mu_p"):
    """
    Create complete drift-diffusion system for semiconductor transport.

    Drift-Diffusion Equations:
    -------------------------
    This implements the classical semiconductor device equations:

    1. Poisson: ∇·(ε∇Φ) = -q(p - n + N_D - N_A)
    2. Electron continuity: ∂n/∂t = (1/q)∇·J_n + G_n - R_n
    3. Hole continuity: ∂p/∂t = -(1/q)∇·J_p + G_p - R_p

    with drift-diffusion currents:
    - J_n = q·μ_n·n·∇Φ + q·D_n·∇n
    - J_p = -q·μ_p·p·∇Φ - q·D_p·∇p

    and SRH recombination-generation.

    This forms a coupled nonlinear system solved by Newton's method.
    """
    CreatePE(device, region)
    CreateBernoulli(device, region)
    CreateSRH(device, region)
    CreateECE(device, region, mu_n)
    CreateHCE(device, region, mu_p)


def CreateSiliconDriftDiffusionAtContact(device, region, contact, is_circuit=False):
    """
    Create contact boundary conditions for drift-diffusion simulation.

    Ohmic Contact Boundary Conditions:
    ---------------------------------
    At an ideal ohmic contact, carriers are in thermal equilibrium:

    For n-type: n = n_eq = (N_D + sqrt(N_D² + 4n_i²))/2
    For p-type: p = p_eq = (-N_D + sqrt(N_D² + 4n_i²))/2

    This enforces:
    1. Carrier concentrations fixed to equilibrium values
    2. Current continuity: ∫J·n̂ dS integrated into circuit

    The contact acts as an infinite reservoir/sink for carriers,
    maintaining local equilibrium regardless of bias.
    """
    # Boundary conditions: carriers equal equilibrium values
    contact_electrons_model = (
        "Electrons - ifelse(NetDoping > 0, {0}, n_i^2/{1})".format(
            celec_model, chole_model
        )
    )
    contact_holes_model = "Holes - ifelse(NetDoping < 0, +{1}, +n_i^2/{0})".format(
        celec_model, chole_model
    )
    contact_electrons_name = "{0}nodeelectrons".format(contact)
    contact_holes_name = "{0}nodeholes".format(contact)

    CreateContactNodeModel(
        device, contact, contact_electrons_name, contact_electrons_model
    )
    # Simplified derivative (derivative = 1 for carrier w.r.t. itself)
    CreateContactNodeModel(
        device, contact, "{0}:{1}".format(contact_electrons_name, "Electrons"), "1"
    )

    CreateContactNodeModel(device, contact, contact_holes_name, contact_holes_model)
    CreateContactNodeModel(
        device, contact, "{0}:{1}".format(contact_holes_name, "Holes"), "1"
    )

    # Integrate currents at contact
    if is_circuit:
        contact_equation(
            device=device,
            contact=contact,
            name="ElectronContinuityEquation",
            node_model=contact_electrons_name,
            edge_current_model="ElectronCurrent",
            circuit_node=GetContactBiasName(contact),
        )

        contact_equation(
            device=device,
            contact=contact,
            name="HoleContinuityEquation",
            node_model=contact_holes_name,
            edge_current_model="HoleCurrent",
            circuit_node=GetContactBiasName(contact),
        )

    else:
        contact_equation(
            device=device,
            contact=contact,
            name="ElectronContinuityEquation",
            node_model=contact_electrons_name,
            edge_current_model="ElectronCurrent",
        )

        contact_equation(
            device=device,
            contact=contact,
            name="HoleContinuityEquation",
            node_model=contact_holes_name,
            edge_current_model="HoleCurrent",
        )


def CreateOxidePotentialOnly(device, region, update_type="default"):
    """
    Create Poisson equation for oxide (insulator) region.

    Oxide Electrostatics:
    --------------------
    In the oxide (SiO₂), there are no mobile carriers, so Poisson
    equation reduces to Laplace equation in neutral regions:

    ∇·(ε_ox·∇Φ) = -ρ_fixed

    where ρ_fixed includes only fixed charges (typically zero in
    high-quality oxides).

    The oxide acts as a dielectric, supporting electric field but
    not conducting current. The displacement field D = ε_ox·E is
    continuous across interfaces.

    Typical oxide permittivity: ε_ox = 3.9·ε₀
    """
    if not InNodeModelList(device, region, "Potential"):
        print("Creating Node Solution Potential")
        CreateSolution(device, region, "Potential")

    # Electric field: E = -∇Φ
    efield = "(Potential@n0 - Potential@n1)*EdgeInverseLength"
    CreateEdgeModel(device, region, "ElectricField", efield)
    CreateEdgeModelDerivatives(device, region, "ElectricField", efield, "Potential")

    # Displacement field: D = ε·E
    dfield = "Permittivity*ElectricField"
    CreateEdgeModel(device, region, "PotentialEdgeFlux", dfield)
    CreateEdgeModelDerivatives(device, region, "PotentialEdgeFlux", dfield, "Potential")

    # Laplace/Poisson equation in oxide
    equation(
        device=device,
        region=region,
        name="PotentialEquation",
        variable_name="Potential",
        edge_model="PotentialEdgeFlux",
        variable_update=update_type,
    )


def CreateSiliconOxideInterface(device, interface):
    """
    Create interface conditions for Si/SiO₂ boundary.

    Silicon-Oxide Interface:
    -----------------------
    At the Si/SiO₂ interface, the following conditions apply:

    1. Potential continuity: Φ_Si = Φ_ox
       (No voltage drop across ideal interface)

    2. Displacement continuity: ε_Si·E_Si⊥ = ε_ox·E_ox⊥
       (Gauss's law with no interface charge)

    3. No carrier flow into oxide: J_n⊥ = J_p⊥ = 0
       (Oxide is insulating)

    The displacement discontinuity arises from the permittivity
    difference: ε_Si/ε_ox ≈ 11.1/3.9 ≈ 2.85
    """
    model_name = CreateContinuousInterfaceModel(device, interface, "Potential")
    interface_equation(
        device=device,
        interface=interface,
        name="PotentialEquation",
        interface_model=model_name,
        type="continuous",
    )


def CreateSiliconSiliconInterface(device, interface):
    """
    Create interface conditions for Si/Si boundary (heterojunction).

    Silicon-Silicon Interface:
    -------------------------
    At a Si/Si interface (e.g., between differently doped regions or
    at a physical heterojunction), continuity is enforced for:

    1. Electrostatic potential: Φ₁ = Φ₂
       (Or with band offset: Φ₁ - Φ₂ = ΔE_c/q for heterojunctions)

    2. Electron concentration: n₁ = n₂
       (Quasi-Fermi level continuity: φ_n₁ = φ_n₂)
       where φ_n = Φ - V_t·ln(n/n_i)

    3. Hole concentration: p₁ = p₂
       (Quasi-Fermi level continuity: φ_p₁ = φ_p₂)
       where φ_p = Φ + V_t·ln(p/n_i)

    4. Current continuity: J_n⊥₁ = J_n⊥₂ and J_p⊥₁ = J_p⊥₂
       (No carrier accumulation at interface)

    These conditions ensure that:
    - No voltage drop occurs across the interface
    - Carrier flow is continuous (no trapping/generation)
    - The quasi-Fermi levels are continuous

    This is appropriate for:
    - Homojunctions (p-n junctions in same material)
    - Ideal abrupt interfaces without interface states
    - Grain boundaries with perfect lattice matching

    For non-ideal interfaces, additional terms for:
    - Interface recombination velocity
    - Interface trapped charge
    - Thermionic emission
    may be required.
    """
    # Enforce potential continuity across interface
    CreateSiliconOxideInterface(device, interface)

    # Enforce electron concentration continuity
    # This implicitly ensures quasi-Fermi level continuity
    ename = CreateContinuousInterfaceModel(device, interface, "Electrons")
    interface_equation(
        device=device,
        interface=interface,
        name="ElectronContinuityEquation",
        interface_model=ename,
        type="continuous",
    )

    # Enforce hole concentration continuity
    # This implicitly ensures quasi-Fermi level continuity
    hname = CreateContinuousInterfaceModel(device, interface, "Holes")
    interface_equation(
        device=device,
        interface=interface,
        name="HoleContinuityEquation",
        interface_model=hname,
        type="continuous",
    )