# demo_0_1d_poisson.py
# Stage 1: Self-consistent Poisson solve for a 1D PN junction in DEVSIM.
# Key correctness points:
#   • add_1d_region: uses tag1/tag2   (NOT start/end)
#   • add_1d_contact: uses tag + material (NOT region / location)
#   • set_node_values uses values= (plural)
#   • node_solution is created BEFORE setting values
#   • Poisson RHS has correct Potential derivative for Newton
#   • Contact BCs via contact_node_model + contact_equation

from devsim import *

device = "diode1d"
region = "Si"
mesh   = "diode_mesh"

# ----------------------------
# 1) Mesh, region, contacts (1D API)
# ----------------------------
create_1d_mesh(mesh=mesh)
add_1d_mesh_line(mesh=mesh, pos=0.0,   ps=2.0, tag="anode_pos")
add_1d_mesh_line(mesh=mesh, pos=48.0,  ps=2.0, tag="p_bulk")
add_1d_mesh_line(mesh=mesh, pos=50.0,  ps=0.2, tag="junction")
add_1d_mesh_line(mesh=mesh, pos=52.0,  ps=2.0, tag="n_bulk")
add_1d_mesh_line(mesh=mesh, pos=100.0, ps=2.0, tag="cathode_pos")

# Region uses tags at its ends (NOT numeric start/end in 1D)
add_1d_region(mesh=mesh, material="Silicon", region=region,
              tag1="anode_pos", tag2="cathode_pos")

# Contacts in 1D: tag + material (no 'region=' / no 'location=' here)
add_1d_contact(mesh=mesh, name="Anode",   material="Silicon", tag="anode_pos")
add_1d_contact(mesh=mesh, name="Cathode", material="Silicon", tag="cathode_pos")

finalize_mesh(mesh=mesh)
create_device(mesh=mesh, device=device)

# ----------------------------
# 2) Parameters (constants & material)
# ----------------------------
# Units: assume geometry is in cm; you can scale as needed
q    = 1.60217662e-19
epss = 11.7 * 8.854187817e-14  # F/cm (Si permittivity)
Vt   = 0.025851                # ~kT/q at 300 K
ni   = 1.0e10                  # intrinsic density (cm^-3)

set_parameter(device=device,               name="ElectronCharge", value=q)
set_parameter(device=device,               name="Permittivity",   value=epss)
set_parameter(device=device,               name="V_t",            value=Vt)
set_parameter(device=device, region=region, name="n_i",           value=ni)

# ----------------------------
# 3) Doping & solution initialization
# ----------------------------
# NetDoping: acceptors (negative) on the left, donors (positive) on the right
NA = 1e17
ND = 1e17

node_model(device=device, region=region, name="NetDoping",
           equation="(x < 50.0) ? -{NA} : {ND}".format(NA=NA, ND=ND))

# Create the Potential solution BEFORE assigning values
node_solution(device=device, region=region, name="Potential")
set_node_values(device=device, region=region, name="Potential", values=0.0)

# ----------------------------
# 4) Carrier models (Boltzmann, equilibrium) and Poisson models
# ----------------------------
# Equilibrium carrier densities as functions of Potential:
#   n = n_i * exp(+Potential / Vt)
#   p = n_i * exp(-Potential / Vt)
# (Reference electrostatic potential taken at 0 V; you can add offsets if desired.)
node_model(device=device, region=region, name="Electrons",
           equation="n_i*exp(Potential/V_t)")
node_model(device=device, region=region, name="Holes",
           equation="n_i*exp(-Potential/V_t)")

# Poisson RHS (node charge):  -q * (p - n + NetDoping)
node_model(device=device, region=region, name="PotentialNodeCharge",
           equation="-ElectronCharge*(Holes - Electrons + NetDoping)")

# d/d(Potential) of the RHS for Newton:
#   dHoles/dV = -(Holes)/Vt, dElectrons/dV = +(Electrons)/Vt
#   d/dV (H - E + D) = ( -H/Vt - E/Vt ) = -(H + E)/Vt
#   => d/dV [ -q*(H - E + D) ] = +q*(H + E)/Vt
node_model(device=device, region=region, name="PotentialNodeCharge:Potential",
           equation="ElectronCharge*(Holes + Electrons)/V_t")

# Electric field on edges and flux for Poisson LHS
edge_model(device=device, region=region, name="ElectricField",
           equation="-Gradient(Potential)")
edge_model(device=device, region=region, name="ElectricField:Potential",
           equation="-EdgeInverseLength")

edge_model(device=device, region=region, name="PotentialEdgeFlux",
           equation="Permittivity*ElectricField")
edge_model(device=device, region=region, name="PotentialEdgeFlux:Potential",
           equation="Permittivity*ElectricField:Potential")

# Assemble Poisson equation
equation(device=device, region=region, name="PotentialEquation",
         variable_name="Potential",
         node_model="PotentialNodeCharge",
         edge_model="PotentialEdgeFlux",
         variable_update="log_damp")

# ----------------------------
# 5) Contacts (Dirichlet BCs for Potential)
# ----------------------------
set_parameter(device=device, name="Anode_bias",   value=0.0)
set_parameter(device=device, name="Cathode_bias", value=0.0)

contact_node_model(device=device, contact="Anode",   name="PotentialBias",
                   equation="Anode_bias")
contact_node_model(device=device, contact="Cathode", name="PotentialBias",
                   equation="Cathode_bias")

contact_equation(device=device, contact="Anode",   name="PotentialEquation",
                 node_model="PotentialBias")
contact_equation(device=device, contact="Cathode", name="PotentialEquation",
                 node_model="PotentialBias")

# ----------------------------
# 6) Solve and simple bias sweep
# ----------------------------
# DC operating point (equilibrium)
solve(type="dc", absolute_error=1e10, relative_error=1e-10, maximum_iterations=50)

print("\nPoisson-only reverse-bias sweep (Potential BC at Cathode):")
for v in [0.0, -0.5, -1.0, -2.0, -5.0]:
    set_parameter(device=device, name="Cathode_bias", value=v)
    solve(type="dc", absolute_error=1e10, relative_error=1e-10, maximum_iterations=50)

    # For Poisson-only, there are no continuity equations;
    # we can still compute a displacement-based 'charge flow' proxy via contact charge.
    # Many users prefer to examine total charge at the contact:
    q_anode = get_contact_charge(device=device, contact="Anode")
    q_cath  = get_contact_charge(device=device, contact="Cathode")
    print(f"V = {v:+.2f} V, Contact charge: Anode={q_anode:.4e} C, Cathode={q_cath:.4e} C")

print("\nDone.")
