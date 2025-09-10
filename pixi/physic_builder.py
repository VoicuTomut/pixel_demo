# ============================================================================
# FILE: physic_builder.py
# PURPOSE: Build complete semiconductor physics model for the device
# ============================================================================

import devsim


def build_physic_model(device_name, photon_flux, alpha, thermal_voltage=0.0259):
    """
    Builds the complete physics model for photodiode simulation.

    This includes:
    1. Electrostatic models (Poisson equation)
    2. Current transport models (drift-diffusion)
    3. Recombination models (SRH)
    4. Optical generation models
    5. All necessary derivatives for Newton solver

    Args:
        device_name (str): Name of the device
        photon_flux (float): Incident photon flux (photons/cm²/s)
        alpha (float): Absorption coefficient (1/cm)
        thermal_voltage (float): Thermal voltage kT/q (V), default 0.0259V at 300K
    """
    print("  3B: Defining all bulk physical models...")

    # ===== SET GLOBAL PARAMETERS =====
    devsim.set_parameter(name="ThermalVoltage", value=thermal_voltage)
    devsim.set_parameter(name="PhotonFlux", value=photon_flux)
    devsim.set_parameter(name="alpha", value=alpha)

    # Apply physics to both regions
    for region in ["p_region", "n_region"]:

        # ===== ELECTROSTATIC MODELS =====
        # These solve Poisson's equation: ∇·(ε∇φ) = -ρ

        # Create edge models from node values for gradient calculations
        devsim.edge_from_node_model(device=device_name, region=region, node_model="Potential")

        # Electric field: E = -∇φ (using finite difference on edges)
        devsim.edge_model(
            device=device_name, region=region, name="ElectricField",
            equation="(Potential@n0 - Potential@n1) * EdgeInverseLength"
        )

        # Displacement field: D = εE
        devsim.edge_model(
            device=device_name, region=region, name="DField",
            equation="Permittivity * ElectricField"
        )

        # Derivatives of D-field for Jacobian matrix
        devsim.edge_model(
            device=device_name, region=region, name="DField:Potential@n0",
            equation="Permittivity * EdgeInverseLength"
        )
        devsim.edge_model(
            device=device_name, region=region, name="DField:Potential@n1",
            equation="-Permittivity * EdgeInverseLength"
        )

        # Space charge density: ρ = q(p - n + N_D - N_A)
        devsim.node_model(
            device=device_name, region=region, name="SpaceCharge",
            equation="ElectronCharge * (Holes - Electrons + NetDoping)"
        )

        # ===== CURRENT TRANSPORT MODELS =====
        # Drift-diffusion equations with Scharfetter-Gummel discretization

        # Voltage difference between nodes (normalized by thermal voltage)
        devsim.edge_model(
            device=device_name, region=region, name="vdiff",
            equation="(Potential@n0 - Potential@n1)/ThermalVoltage"
        )

        # Bernoulli function for stable discretization
        # B(x) = x/(exp(x)-1), handles both drift and diffusion
        devsim.edge_model(
            device=device_name, region=region, name="Bernoulli_vdiff",
            equation="B(vdiff)"
        )
        devsim.edge_model(
            device=device_name, region=region, name="Bernoulli_neg_vdiff",
            equation="B(-vdiff)"
        )

        # Create edge models for carrier concentrations
        devsim.edge_from_node_model(device=device_name, region=region, node_model="Electrons")
        devsim.edge_from_node_model(device=device_name, region=region, node_model="Holes")

        # Create edge models for mobility (average between nodes)
        devsim.edge_from_node_model(device=device_name, region=region, node_model="ElectronMobility")
        devsim.edge_from_node_model(device=device_name, region=region, node_model="HoleMobility")

        devsim.edge_model(
            device=device_name, region=region, name="EdgeElectronMobility",
            equation="(ElectronMobility@n0 + ElectronMobility@n1) * 0.5"
        )
        devsim.edge_model(
            device=device_name, region=region, name="EdgeHoleMobility",
            equation="(HoleMobility@n0 + HoleMobility@n1) * 0.5"
        )

        # Electron current density: J_n = qμ_n n∇φ + qD_n∇n
        # Using Scharfetter-Gummel discretization
        electron_current_eq = (
            "ElectronCharge * EdgeElectronMobility * ThermalVoltage * EdgeInverseLength * "
            "(Electrons@n1 * Bernoulli_neg_vdiff - Electrons@n0 * Bernoulli_vdiff)"
        )
        devsim.edge_model(
            device=device_name, region=region, name="ElectronCurrent",
            equation=electron_current_eq
        )

        # Derivatives of electron current for Jacobian
        for v in ["Potential", "Electrons", "Holes"]:
            for n in ["n0", "n1"]:
                devsim.edge_model(
                    device=device_name, region=region,
                    name=f"ElectronCurrent:{v}@{n}",
                    equation=f"diff({electron_current_eq}, {v}@{n})"
                )

        # Hole current density: J_p = qμ_p p∇φ - qD_p∇p
        # Note opposite sign convention vs electrons
        hole_current_eq = (
            "ElectronCharge * EdgeHoleMobility * ThermalVoltage * EdgeInverseLength * "
            "(Holes@n1 * Bernoulli_vdiff - Holes@n0 * Bernoulli_neg_vdiff)"
        )
        devsim.edge_model(
            device=device_name, region=region, name="HoleCurrent",
            equation=hole_current_eq
        )

        # Derivatives of hole current for Jacobian
        for v in ["Potential", "Electrons", "Holes"]:
            for n in ["n0", "n1"]:
                devsim.edge_model(
                    device=device_name, region=region,
                    name=f"HoleCurrent:{v}@{n}",
                    equation=f"diff({hole_current_eq}, {v}@{n})"
                )

        # ===== CARRIER STATISTICS =====
        # Calculate intrinsic carrier concentrations for equilibrium

        # Intrinsic carrier concentration squared
        devsim.node_model(
            device=device_name, region=region, name="n_i_squared",
            equation="IntrinsicCarrierDensity^2"
        )

        # Intrinsic electron concentration (from charge neutrality)
        devsim.node_model(
            device=device_name, region=region, name="IntrinsicElectrons",
            equation="0.5*(NetDoping+(NetDoping^2+4*n_i_squared)^0.5)"
        )

        # Intrinsic hole concentration (from charge neutrality)
        devsim.node_model(
            device=device_name, region=region, name="IntrinsicHoles",
            equation="0.5*(-NetDoping+(NetDoping^2+4*n_i_squared)^0.5)"
        )

        # ===== RECOMBINATION MODELS =====

        # Shockley-Read-Hall (SRH) recombination
        # R_SRH = (np - n_i²) / (τ_p(n + n_i) + τ_n(p + p_i))
        srh_eq = (
            "(Electrons*Holes - n_i_squared) / "
            "(taup*(Electrons + IntrinsicElectrons) + taun*(Holes + IntrinsicHoles))"
        )
        devsim.node_model(
            device=device_name, region=region, name="USRH",
            equation=srh_eq
        )

        # ===== OPTICAL GENERATION MODEL =====
        # Beer-Lambert law: G(y) = Φα exp(-αy)
        # Factor of 0.5 for 2D simulation normalization
        devsim.node_model(
            device=device_name, region=region, name="OpticalGeneration",
            equation="PhotonFlux * alpha * exp(-alpha * abs(y))*0.5"
        )

        # ===== NET RECOMBINATION =====
        # Net recombination rate: R_net = R_SRH - G_optical
        net_recombination_eq = "USRH - OpticalGeneration"

        devsim.node_model(
            device=device_name, region=region, name="NetRecombination",
            equation=net_recombination_eq
        )

        # Derivatives of net recombination for Jacobian
        devsim.node_model(
            device=device_name, region=region, name="NetRecombination:Electrons",
            equation=f"diff({net_recombination_eq}, Electrons)"
        )
        devsim.node_model(
            device=device_name, region=region, name="NetRecombination:Holes",
            equation=f"diff({net_recombination_eq}, Holes)"
        )

        # ===== SOURCE TERMS FOR CONTINUITY EQUATIONS =====
        # Source term for electron continuity: S_n = qR
        devsim.node_model(
            device=device_name, region=region, name="eCharge_x_NetRecomb",
            equation="ElectronCharge * NetRecombination"
        )
        devsim.node_model(
            device=device_name, region=region, name="eCharge_x_NetRecomb:Electrons",
            equation="ElectronCharge * NetRecombination:Electrons"
        )
        devsim.node_model(
            device=device_name, region=region, name="eCharge_x_NetRecomb:Holes",
            equation="ElectronCharge * NetRecombination:Holes"
        )

        # Source term for hole continuity: S_p = -qR
        devsim.node_model(
            device=device_name, region=region, name="Neg_eCharge_x_NetRecomb",
            equation="-ElectronCharge * NetRecombination"
        )
        devsim.node_model(
            device=device_name, region=region, name="Neg_eCharge_x_NetRecomb:Electrons",
            equation="-ElectronCharge * NetRecombination:Electrons"
        )
        devsim.node_model(
            device=device_name, region=region, name="Neg_eCharge_x_NetRecomb:Holes",
            equation="-ElectronCharge * NetRecombination:Holes"
        )

