import devsim

def define_doping(device, p_doping, n_doping):
    """Defines the acceptor and donor concentrations for the device."""
    devsim.node_model(device=device, region="p_region", name="Acceptors", equation=f"{p_doping}")
    devsim.node_model(device=device, region="p_region", name="Donors", equation="0.0")
    devsim.node_model(device=device, region="n_region", name="Acceptors", equation="0.0")
    devsim.node_model(device=device, region="n_region", name="Donors", equation=f"{n_doping}")
    for region in ["p_region", "n_region"]:
        devsim.node_model(device=device, region=region, name="NetDoping", equation="Donors - Acceptors")
    print(f"Defined doping: N_A = {p_doping:.1e} cm^-3, N_D = {n_doping:.1e} cm^-3")

