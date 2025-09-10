import devsim
def solve_equilibrium(device_name):
    print("  3D: Solving for initial equilibrium state (two-step method)...")
    print("    Creating robust initial guess based on charge neutrality...")
    for region in ["p_region", "n_region"]:
        devsim.set_node_values(device=device_name, region=region, name="Electrons", init_from="IntrinsicElectrons")
        devsim.set_node_values(device=device_name, region=region, name="Holes", init_from="IntrinsicHoles")
        devsim.node_model(device=device_name, region=region, name="InitialPotential",
                          equation="ThermalVoltage * log(IntrinsicElectrons/IntrinsicCarrierDensity)")
        devsim.set_node_values(device=device_name, region=region, name="Potential", init_from="InitialPotential")

    print("    Step 1/2: Assembling and solving for Potential only...")
    # FIRST: Setup Potential equation in regions
    for region in ["p_region", "n_region"]:
        devsim.equation(device=device_name, region=region, name="PotentialEquation",
                        variable_name="Potential",
                        node_model="SpaceCharge",  # <-- THIS IS THE CRITICAL FIX
                        edge_model="DField",
                        variable_update="log_damp")

    # Setup contact equations for Potential (without edge_charge_model initially)
    for contact in ["anode", "cathode"]:
        devsim.contact_equation(device=device_name, contact=contact, name="PotentialEquation",
                                node_model=f"{contact}_potential_bc")

    # Setup interface equation for Potential
    devsim.interface_equation(device=device_name, interface="pn_junction", name="PotentialEquation",
                              interface_model="Potential_continuity", type="continuous")

    # Solve for Potential only
    devsim.solve(type="dc", absolute_error=1e-10, relative_error=1e-12, maximum_iterations=50)

    print("    Updating carrier guess using Boltzmann statistics...")
    for region in ["p_region", "n_region"]:
        devsim.node_model(device=device_name, region=region, name="UpdatedElectrons",
                          equation="IntrinsicCarrierDensity*exp(Potential/ThermalVoltage)")
        devsim.node_model(device=device_name, region=region, name="UpdatedHoles",
                          equation="IntrinsicCarrierDensity*exp(-Potential/ThermalVoltage)")
        devsim.set_node_values(device=device_name, region=region, name="Electrons", init_from="UpdatedElectrons")
        devsim.set_node_values(device=device_name, region=region, name="Holes", init_from="UpdatedHoles")

    print("    Step 2/2: Assembling continuity equations and solving the full system...")
    # SECOND: Setup continuity equations in regions
    for region in ["p_region", "n_region"]:
        # For d(n)/dt: div(Jn) = q(R-G). DEVSIM solves div(F) - S = 0.
        # So F=Jn and S = q(R-G), which is our "eCharge_x_NetRecomb"
        devsim.equation(device=device_name, region=region, name="ElectronContinuityEquation",
                        variable_name="Electrons",
                        node_model="eCharge_x_NetRecomb",
                        edge_model="ElectronCurrent",
                        variable_update="log_damp")

        # For d(p)/dt: div(Jp) = -q(R-G). DEVSIM solves div(F) - S = 0.
        # So F=Jp and S = -q(R-G), which is our "Neg_eCharge_x_NetRecomb"
        devsim.equation(device=device_name, region=region, name="HoleContinuityEquation",
                        variable_name="Holes",
                        node_model="Neg_eCharge_x_NetRecomb",
                        edge_model="HoleCurrent",
                        variable_update="log_damp")

    # Setup interface equations for continuity
    devsim.interface_equation(device=device_name, interface="pn_junction", name="ElectronContinuityEquation",
                              interface_model="Electrons_continuity", type="continuous")
    devsim.interface_equation(device=device_name, interface="pn_junction", name="HoleContinuityEquation",
                              interface_model="Holes_continuity", type="continuous")

    # CRITICAL: Setup ALL contact equations WITH edge_charge_model for Potential
    for contact in ["anode", "cathode"]:
        devsim.contact_equation(device=device_name, contact=contact, name="PotentialEquation",
                                node_model=f"{contact}_potential_bc",
                                edge_charge_model="DField")  # THIS IS CRITICAL FOR C-V
        devsim.contact_equation(device=device_name, contact=contact, name="ElectronContinuityEquation",
                                node_model=f"{contact}_electrons_bc",
                                edge_current_model="ElectronCurrent")
        devsim.contact_equation(device=device_name, contact=contact, name="HoleContinuityEquation",
                                node_model=f"{contact}_holes_bc",
                                edge_current_model="HoleCurrent")