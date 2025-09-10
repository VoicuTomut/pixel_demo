
import devsim
def set_boundary(device_name):
    print("  3C: Defining all boundary condition models...")
    for contact in ["anode", "cathode"]:
        devsim.set_parameter(device=device_name, name=f"{contact}_bias", value=0.0)
        # Use f-strings to create unique names like "anode_potential_bc"
        devsim.contact_node_model(device=device_name, contact=contact, name=f"{contact}_potential_bc",
                                  equation=f"Potential - {contact}_bias")
        devsim.contact_node_model(device=device_name, contact=contact, name=f"{contact}_potential_bc:Potential",
                                  equation="1.0")
        devsim.contact_node_model(device=device_name, contact=contact, name=f"{contact}_electrons_bc",
                                  equation="Electrons - IntrinsicElectrons")
        devsim.contact_node_model(device=device_name, contact=contact, name=f"{contact}_electrons_bc:Electrons",
                                  equation="1.0")
        devsim.contact_node_model(device=device_name, contact=contact, name=f"{contact}_holes_bc",
                                  equation="Holes - IntrinsicHoles")
        devsim.contact_node_model(device=device_name, contact=contact, name=f"{contact}_holes_bc:Holes", equation="1.0")

    for variable in ["Potential", "Electrons", "Holes"]:
        devsim.interface_model(device=device_name, interface="pn_junction", name=f"{variable}_continuity",
                               equation=f"{variable}@r0 - {variable}@r1")
        devsim.interface_model(device=device_name, interface="pn_junction", name=f"{variable}_continuity:{variable}@r0",
                               equation="1.0")
        devsim.interface_model(device=device_name, interface="pn_junction", name=f"{variable}_continuity:{variable}@r1",
                               equation="-1.0")