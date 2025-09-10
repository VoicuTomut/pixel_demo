import devsim

def build_physic_model(device_name, photon_flux, alpha, thermal_voltage=0.0259):
    print("  3B: Defining all bulk physical models...")
    devsim.set_parameter(name="ThermalVoltage", value=thermal_voltage)
    devsim.set_parameter(name="PhotonFlux", value=photon_flux)
    devsim.set_parameter(name="alpha", value=alpha)

    for region in ["p_region", "n_region"]:
        devsim.edge_from_node_model(device=device_name, region=region, node_model="Potential")
        devsim.edge_model(device=device_name, region=region, name="ElectricField",
                          equation="(Potential@n0 - Potential@n1) * EdgeInverseLength")
        devsim.edge_model(device=device_name, region=region, name="DField", equation="Permittivity * ElectricField")
        devsim.edge_model(device=device_name, region=region, name="DField:Potential@n0",
                          equation="Permittivity * EdgeInverseLength")
        devsim.edge_model(device=device_name, region=region, name="DField:Potential@n1",
                          equation="-Permittivity * EdgeInverseLength")
        devsim.node_model(device=device_name, region=region, name="SpaceCharge",
                          equation="ElectronCharge * (Holes - Electrons + NetDoping)")
        devsim.edge_model(device=device_name, region=region, name="vdiff",
                          equation="(Potential@n0 - Potential@n1)/ThermalVoltage")
        devsim.edge_model(device=device_name, region=region, name="Bernoulli_vdiff", equation="B(vdiff)")
        devsim.edge_model(device=device_name, region=region, name="Bernoulli_neg_vdiff", equation="B(-vdiff)")
        devsim.edge_from_node_model(device=device_name, region=region, node_model="Electrons")
        devsim.edge_from_node_model(device=device_name, region=region, node_model="Holes")
        devsim.edge_from_node_model(device=device_name, region=region, node_model="ElectronMobility")
        devsim.edge_from_node_model(device=device_name, region=region,
                                    node_model="HoleMobility")  # Typo was corrected here
        devsim.edge_model(device=device_name, region=region, name="EdgeElectronMobility",
                          equation="(ElectronMobility@n0 + ElectronMobility@n1) * 0.5")
        devsim.edge_model(device=device_name, region=region, name="EdgeHoleMobility",
                          equation="(HoleMobility@n0 + HoleMobility@n1) * 0.5")
        electron_current_eq = "ElectronCharge * EdgeElectronMobility * ThermalVoltage * EdgeInverseLength * (Electrons@n1 * Bernoulli_neg_vdiff - Electrons@n0 * Bernoulli_vdiff)"
        devsim.edge_model(device=device_name, region=region, name="ElectronCurrent", equation=electron_current_eq)
        for v in ["Potential", "Electrons", "Holes"]:
            for n in ["n0", "n1"]: devsim.edge_model(device=device_name, region=region, name=f"ElectronCurrent:{v}@{n}",
                                                     equation=f"diff({electron_current_eq}, {v}@{n})")
        hole_current_eq = "ElectronCharge * EdgeHoleMobility * ThermalVoltage * EdgeInverseLength * (Holes@n1 * Bernoulli_vdiff - Holes@n0 * Bernoulli_neg_vdiff)"
        devsim.edge_model(device=device_name, region=region, name="HoleCurrent", equation=hole_current_eq)
        for v in ["Potential", "Electrons", "Holes"]:
            for n in ["n0", "n1"]: devsim.edge_model(device=device_name, region=region, name=f"HoleCurrent:{v}@{n}",
                                                     equation=f"diff({hole_current_eq}, {v}@{n})")

        devsim.node_model(device=device_name, region=region, name="n_i_squared", equation="IntrinsicCarrierDensity^2")
        devsim.node_model(device=device_name, region=region, name="IntrinsicElectrons",
                          equation="0.5*(NetDoping+(NetDoping^2+4*n_i_squared)^0.5)")
        devsim.node_model(device=device_name, region=region, name="IntrinsicHoles",
                          equation="0.5*(-NetDoping+(NetDoping^2+4*n_i_squared)^0.5)")
        srh_eq = "(Electrons*Holes - n_i_squared) / (taup*(Electrons + IntrinsicElectrons) + taun*(Holes + IntrinsicHoles))"
        devsim.node_model(device=device_name, region=region, name="USRH", equation=srh_eq)
        # Use abs(y) for robustness, though (0-y) is also fine.
        devsim.node_model(device=device_name, region=region, name="OpticalGeneration",
                          equation="PhotonFlux * alpha * exp(-alpha * abs(y))*0.5")

        # Define the full NetRecombination equation in a single string for consistency
        net_recombination_eq = "USRH - OpticalGeneration"

        # Use the full string to define the model
        devsim.node_model(device=device_name, region=region, name="NetRecombination", equation=net_recombination_eq)

        # CRITICAL: Use the full string to define the derivatives
        devsim.node_model(device=device_name, region=region, name="NetRecombination:Electrons",
                          equation=f"diff({net_recombination_eq}, Electrons)")
        devsim.node_model(device=device_name, region=region, name="NetRecombination:Holes",
                          equation=f"diff({net_recombination_eq}, Holes)")
        devsim.node_model(device=device_name, region=region, name="eCharge_x_NetRecomb",
                          equation="ElectronCharge * NetRecombination")
        devsim.node_model(device=device_name, region=region, name="eCharge_x_NetRecomb:Electrons",
                          equation="ElectronCharge * NetRecombination:Electrons")
        devsim.node_model(device=device_name, region=region, name="eCharge_x_NetRecomb:Holes",
                          equation="ElectronCharge * NetRecombination:Holes")

        devsim.node_model(device=device_name, region=region, name="Neg_eCharge_x_NetRecomb",
                          equation="-ElectronCharge * NetRecombination")
        devsim.node_model(device=device_name, region=region, name="Neg_eCharge_x_NetRecomb:Electrons",
                          equation="-ElectronCharge * NetRecombination:Electrons")
        devsim.node_model(device=device_name, region=region, name="Neg_eCharge_x_NetRecomb:Holes",
                          equation="-ElectronCharge * NetRecombination:Holes")

