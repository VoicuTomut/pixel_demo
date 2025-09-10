import devsim

def define_mobility_models(device, region):
    """
    Defines doping-dependent mobility using the Caughey-Thomas model.
    """
    devsim.node_model(device=device, region=region, name="TotalDoping", equation="abs(Acceptors) + abs(Donors)")
    mu_max_n, mu_min_n, N_ref_n, alpha_n = 1417.0, 68.5, 1.10e17, 0.711
    eqn_n = f"{mu_min_n} + ({mu_max_n} - {mu_min_n}) / (1 + (TotalDoping / {N_ref_n})^{alpha_n})"
    devsim.node_model(device=device, region=region, name="ElectronMobility", equation=eqn_n)
    mu_max_p, mu_min_p, N_ref_p, alpha_p = 470.5, 44.9, 2.23e17, 0.719
    eqn_p = f"{mu_min_p} + ({mu_max_p} - {mu_min_p}) / (1 + (TotalDoping / {N_ref_p})^{alpha_p})"
    devsim.node_model(device=device, region=region, name="HoleMobility", equation=eqn_p)
