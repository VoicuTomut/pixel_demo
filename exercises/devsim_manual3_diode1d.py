# devsim_manual3_diode1d_enhanced.py
#
# This script simulates a simple 1D Silicon p-n diode.
# It has been enhanced with detailed comments explaining the simulation
# steps and the underlying semiconductor physics.
#
# The simulation process involves:
# 1. MESHING: Discretizing the 1D device into a series of points (nodes).
# 2. DEFINITION: Setting material parameters (Silicon) and doping profiles.
# 3. EQUILIBRIUM SOLUTION: Solving Poisson's equation to find the initial state
#    with zero applied voltage.
# 4. DRIFT-DIFFUSION: Solving the fully coupled set of semiconductor equations
#    (Poisson, Electron Continuity, Hole Continuity).
# 5. BIAS RAMP: Applying a forward voltage incrementally and solving at each
#    step to trace the I-V curve.
# 6. VISUALIZATION: Plotting the results.

from devsim import *
from devsim.python_packages.simple_physics import *
import matplotlib.pyplot as plt
import numpy as np

# Define convenient names for the device and its single region
device = "MyDevice"
region = "MyRegion"

# --------------------------------------------------------------------
## MESH AND DEVICE CREATION
# --------------------------------------------------------------------
# A mesh is a discretization of the device geometry. We define a 1D line
# and specify the spacing between points. Finer spacing is used where
# physical quantities change rapidly, like at the p-n junction.
create_1d_mesh(mesh="dio")
# Position 'pos' is in cm, spacing 'ps' is in cm.
# We create a 1 micron (1e-4 cm) long device.
add_1d_mesh_line(mesh="dio", pos=0, ps=1e-7, tag="top")  # Top contact
add_1d_mesh_line(mesh="dio", pos=0.5e-5, ps=1e-9, tag="mid")  # Junction (finer mesh)
add_1d_mesh_line(mesh="dio", pos=1e-5, ps=1e-7, tag="bot")  # Bottom contact

# Define the electrical contacts at the ends of the device.
add_1d_contact(mesh="dio", name="top", tag="top", material="metal")
add_1d_contact(mesh="dio", name="bot", tag="bot", material="metal")

# Define the semiconductor region between the contacts.
add_1d_region(mesh="dio", material="Si", region=region, tag1="top", tag2="bot")

# Finalize the mesh and create the device structure for the simulator.
finalize_mesh(mesh="dio")
create_device(mesh="dio", device=device)

# --------------------------------------------------------------------
## SET MATERIAL PARAMETERS AND DOPING PROFILE 🔬
# --------------------------------------------------------------------
# Set standard silicon parameters (permittivity, bandgap, etc.) for a
# temperature of 300 K.
SetSiliconParameters(device, region, 300)

# Set carrier lifetimes. These parameters are crucial for modeling
# recombination via the Shockley-Read-Hall (SRH) mechanism.
# The SRH recombination rate 'U' is given by the general formula:
# U = (p*n - n_i^2) / (tau_p*(n + n1) + tau_n*(p + p1))
#
# Where:
#   p, n:      Hole and electron concentrations.
#   n_i:       Intrinsic carrier concentration.
#   tau_p, tau_n:  Hole and electron minority carrier lifetimes.
#   n1, p1:    Electron and hole concentrations when the Fermi level
#              coincides with the trap energy level (E_t).
#
# The terms n1 and p1 depend on how deep the trap is within the bandgap.
# For the special case of a trap exactly at the mid-gap (E_t = E_i),
# then n1 = p1 = n_i, which simplifies the formula. Your original comment
# used this specific, less general for

set_parameter(device=device, region=region, name="taun", value=1e-8)  # Electron lifetime
set_parameter(device=device, region=region, name="taup", value=1e-8)  # Hole lifetime

# Define the doping profile to create the p-n junction.
# We use a step function to create an abrupt junction at x = 0.5e-5 cm.
#
# NetDoping = N_D - N_A
# where N_D is the donor concentration (n-type) and N_A is the
# acceptor concentration (p-type).

# p-type region: N_A = 1e18 cm^-3 for x < 0.5e-5 cm.
CreateNodeModel(device, region, "Acceptors", "1.0e18*step(0.5e-5-x)")
# n-type region: N_D = 1e18 cm^-3 for x > 0.5e-5 cm.
CreateNodeModel(device, region, "Donors", "1.e18*step(x-0.5e-5)")
# This model calculates the net doping concentration at each node.
CreateNodeModel(device, region, "NetDoping", "Donors - Acceptors")

# --------------------------------------------------------------------
## POTENTIAL-ONLY SOLUTION (EQUILIBRIUM)
# --------------------------------------------------------------------
# This initial step solves ONLY Poisson's equation to find the built-in
# potential and electric field at thermal equilibrium (zero bias).
#
# Poisson's Equation:
# ∇²φ = -(q/ε) * (p - n + N_D⁺ - N_A⁻)
#
# Here, φ is the electrostatic potential, q is elementary charge,
# ε is permittivity, and p, n, N_D⁺, N_A⁻ are the hole, electron,
# ionized donor, and ionized acceptor concentrations, respectively.

CreateSolution(device, region, "Potential")
CreateSiliconPotentialOnly(device, region)

# Set contact voltages to 0V for the equilibrium solve.
for i in get_contact_list(device=device):
    set_parameter(device=device, name=GetContactBiasName(i), value=0.0)
    CreateSiliconPotentialOnlyContact(device, region, i)

# Solve the system.
solve(type="dc", absolute_error=1.0, relative_error=1e-12, maximum_iterations=30)

# --------------------------------------------------------------------
## FULL DRIFT-DIFFUSION SOLUTION
# --------------------------------------------------------------------
# This section activates the full set of semiconductor device physics. We transition
# from a simple electrostatic problem to solving a coupled system of three
# non-linear partial differential equations. These equations self-consistently
# determine the electrostatic potential (φ), electron concentration (n), and
# hole concentration (p) throughout the device.

# The three core equations are:

# 1. Poisson's Equation: Relates the electrostatic potential to the net charge density.
#    The charge density now includes the mobile carriers 'n' and 'p' as solved variables.
#    $$ \nabla^2\phi = -\frac{q}{\epsilon}(p - n + N_D^+ - N_A^-) $$

# 2. Electron Continuity Equation (steady-state): A conservation law for electrons.
#    It states that the net flow of electrons into or out of a point (divergence of
#    electron current density, ∇·J_n) must be balanced by the net recombination/generation
#    rate (U) at that point.
#    $$ \frac{1}{q} \nabla \cdot \mathbf{J_n} = U $$

# 3. Hole Continuity Equation (steady-state): The corresponding conservation law for holes.
#    $$ -\frac{1}{q} \nabla \cdot \mathbf{J_p} = U $$

# The current densities, J_n and J_p, are described by the Drift-Diffusion model,
# which accounts for the two primary mechanisms of charge transport:

#   a) Drift: The movement of charge carriers under the influence of an electric field (E = -∇φ).
#   b) Diffusion: The movement of carriers from a region of high concentration to
#      a region of low concentration, driven by the concentration gradient (∇n or ∇p).

#    Electron Current Density:
#    $$ \mathbf{J_n} = q\mu_n n \mathbf{E} + qD_n \nabla n $$
#    (Drift Term)   (Diffusion Term)

#    Hole Current Density:
#    $$ \mathbf{J_p} = q\mu_p p \mathbf{E} - qD_p \nabla p $$
#    (Drift Term)   (Diffusion Term)

# --- Code Implementation ---

# STEP 1: Declare 'Electrons' (n) and 'Holes' (p) as new unknown variables for the solver.
# The system now has three unknowns to solve for: Potential, Electrons, and Holes.
CreateSolution(device, region, "Electrons")
CreateSolution(device, region, "Holes")

# STEP 2: Provide a good initial guess for the solver.
# A numerical solver needs a starting point. We use the physically-sound results from
# the equilibrium solution (IntrinsicElectrons/Holes) to ensure the solver converges efficiently.
set_node_values(device=device, region=region, name="Electrons", init_from="IntrinsicElectrons")
set_node_values(device=device, region=region, name="Holes", init_from="IntrinsicHoles")

# STEP 3: Define the mathematical relationships between the variables.
# This function loads the Poisson, Electron Continuity, and Hole Continuity equations
# (including the drift-diffusion and recombination models) into the simulator.
CreateSiliconDriftDiffusion(device, region)

# STEP 4: Define the boundary conditions at the metal contacts.
# For each contact, this function sets the rules for how the variables behave at the
# device's edges. It fixes the carrier concentrations to their equilibrium values
# and sets up the mechanism for calculating the current flowing out of the contact.
for i in get_contact_list(device=device):
    CreateSiliconDriftDiffusionAtContact(device, region, i)

# STEP 5: Solve the fully coupled system.
# This command triggers the numerical solver to find the values of φ(x), n(x), and p(x)
# at every mesh point that simultaneously satisfy all three equations and the boundary conditions.
# We solve it first at 0V bias to get a stable starting point for the I-V ramp.
solve(type="dc", absolute_error=1e10, relative_error=1e-10, maximum_iterations=30)

# --------------------------------------------------------------------
## RAMP FORWARD BIAS & COLLECT DATA ⚡️
# --------------------------------------------------------------------
# To get the I-V curve, we apply a forward bias to the 'top' contact
# (the p-side) in small steps. At each voltage step, we re-solve the
# drift-diffusion equations.

# List to store (voltage, current) pairs for the I-V curve
iv_curve_data = []

v = 0.0
while v < 0.51:
    # Apply the forward bias voltage 'v' to the top contact.
    # The bottom contact is implicitly held at 0V (ground).
    set_parameter(device=device, name=GetContactBiasName("top"), value=v)

    # Solve the drift-diffusion system for this new bias condition.
    solve(type="dc", absolute_error=1e10, relative_error=1e-10, maximum_iterations=30)

    # Calculate the total current flowing out of the top contact.
    # Total Current = Electron Current + Hole Current
    j_electron = get_contact_current(device=device, contact="top", equation="ElectronContinuityEquation")
    j_hole = get_contact_current(device=device, contact="top", equation="HoleContinuityEquation")
    total_current = j_electron + j_hole

    # Store the data point. Current is in A/cm for 1D simulations (representing A/cm^2).
    iv_curve_data.append((v, abs(total_current)))

    # Print currents for monitoring progress. At steady state,
    # the current at the top and bottom contacts should be equal and opposite.
    print(f"Voltage: {v:.1f} V")
    PrintCurrents(device, "top")
    PrintCurrents(device, "bot")

    v += 0.1

# --------------------------------------------------------------------
## SAVE RESULTS FILE
# --------------------------------------------------------------------
# Save the final device state (at the last bias point) to a file
# that can be visualized with other tools like Tecplot or VisIt.
write_devices(file="diode_1d.dat", type="tecplot")

#####################################################################
###  PLOTTING SECTION 📈
#####################################################################
#
# Goal: Visualize the simulation results using matplotlib.
# We will plot key physical quantities from the final bias point (V=0.5V).
#
print("\n--- Generating Plots ---")

# --- Get final data from DEVSIM at the last bias point ---
# Node models exist at each mesh point.
# Convert position from cm to micrometers (µm) for better readability.
x_pos_node = np.array(get_node_model_values(device=device, region=region, name="x")) * 1e4
donors = get_node_model_values(device=device, region=region, name="Donors")
acceptors = get_node_model_values(device=device, region=region, name="Acceptors")
electrons = get_node_model_values(device=device, region=region, name="Electrons")
holes = get_node_model_values(device=device, region=region, name="Holes")
potential = get_node_model_values(device=device, region=region, name="Potential")

# Edge models exist on the 'edge' between two nodes.
# Their position is the midpoint between the nodes.
electric_field_edge = get_edge_model_values(device=device, region=region, name="ElectricField")
x_pos_edge = (x_pos_node[:-1] + x_pos_node[1:]) / 2.0

# --- Plot 1: Carrier Concentrations ---
# This plot shows the p-n junction.
# - Left side (x < 0.5 µm) is p-type: hole concentration > electron concentration.
# - Right side (x > 0.5 µm) is n-type: electron concentration > hole concentration.
# - The central area where carrier concentrations drop is the 'depletion region'.
plt.figure(figsize=(10, 6))
plt.plot(x_pos_node, acceptors, color='purple', label="Acceptors ($N_A$)")
plt.plot(x_pos_node, donors, color='green', label="Donors ($N_D$)")
plt.plot(x_pos_node, electrons, color='blue', label="Electrons (n)")
plt.plot(x_pos_node, holes, color='red', label="Holes (p)")
plt.yscale('log')
plt.grid(True, which="both", ls="-", alpha=0.5)
plt.xlabel("Position (µm)", fontsize=14)
plt.ylabel("Concentration (cm$^{-3}$)", fontsize=14)
plt.title("Doping and Carrier Concentrations at V = 0.5 V", fontsize=16)
plt.legend(fontsize=12)
plt.ylim(1e2, 1e19)  # Set y-axis limits for a better view
plt.tight_layout()
plt.savefig("diode_concentrations.png")
print("Saved: diode_concentrations.png")

# --- Plot 2: Potential and Electric Field ---
# This plot shows the potential barrier that charge carriers must overcome.
# The electric field (E = -∇φ) is the negative gradient of the potential.
# It is very strong and points from the n-side to the p-side, confined
# mostly within the depletion region.
fig, ax1 = plt.subplots(figsize=(10, 6))
ax1.plot(x_pos_node, potential, 'b-', label="Potential", linewidth=2)
ax1.set_xlabel("Position (µm)", fontsize=14)
ax1.set_ylabel("Potential (V)", color='b', fontsize=14)
ax1.tick_params(axis='y', labelcolor='b')
ax1.grid(True, which="both", ls="--", alpha=0.5)

# Create a second y-axis that shares the same x-axis for the E-field.
ax2 = ax1.twinx()
ax2.plot(x_pos_edge, electric_field_edge, 'r-', label="Electric Field", linewidth=2)
ax2.set_ylabel("Electric Field (V/cm)", color='r', fontsize=14)
ax2.tick_params(axis='y', labelcolor='r')

plt.title("Potential and Electric Field at V = 0.5 V", fontsize=16)
fig.tight_layout()
plt.savefig("diode_potential_efield.png")
print("Saved: diode_potential_efield.png")

# --- Plot 3: I-V Curve ---
# This plot shows the classic exponential current-voltage relationship of a diode.
# The current is very small for V < ~0.4V and then increases rapidly.
# This behavior is described by the Shockley Diode Equation:
# I = I_0 * (exp(qV / (n*k*T)) - 1)
voltages = [item[0] for item in iv_curve_data]
currents = [item[1] for item in iv_curve_data]

plt.figure(figsize=(8, 6))
plt.plot(voltages, currents, 'o-')
plt.yscale('log')  # A log scale clearly shows the exponential turn-on.
plt.grid(True, which="both", ls="-", alpha=0.5)
plt.xlabel("Voltage (V)", fontsize=14)
plt.ylabel("Current (A/cm)", fontsize=14)
plt.title("Diode I-V Curve", fontsize=16)
plt.tight_layout()
plt.savefig("diode_iv_curve.png")
print("Saved: diode_iv_curve.png")

# --- Plot 4: Recombination and Current Densities ---
# This plot illustrates how current flows through the device.
# - In the p-region (left), current is dominated by holes.
# - In the n-region (right), current is dominated by electrons.
# - In the junction, holes and electrons meet and recombine. The SRH
#   recombination rate peaks here. This recombination is what "converts"
#   hole current into electron current, ensuring current continuity.
fig, ax1 = plt.subplots(figsize=(10, 6))
plt.title("Recombination and Current Density at V = 0.5 V", fontsize=16)

# Get data, using original units (cm) for the x-axis.
x_pos_cm = np.array(get_node_model_values(device=device, region=region, name="x"))
x_pos_edge_cm = (x_pos_cm[:-1] + x_pos_cm[1:]) / 2.0

# USRH (Shockley-Read-Hall recombination) is a node model.
usrh = np.array(get_node_model_values(device=device, region=region, name="USRH"))

# Currents are edge models.
electron_current = np.array(get_edge_model_values(device=device, region=region, name="ElectronCurrent"))
hole_current = np.array(get_edge_model_values(device=device, region=region, name="HoleCurrent"))

# Left Y-axis for Currents
color = 'black'
ax1.set_xlabel("Position (cm)", fontsize=14)
ax1.set_ylabel("Current Density (A/cm$^2$)", color=color, fontsize=14)
# Plot absolute values to see the magnitude of each current component.
p1, = ax1.plot(x_pos_edge_cm, abs(electron_current), color='black', label="ElectronCurrent")
p2, = ax1.plot(x_pos_edge_cm, abs(hole_current), color='blue', label="HoleCurrent")
ax1.tick_params(axis='y', labelcolor=color)
ax1.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

# Right Y-axis for USRH Recombination
ax2 = ax1.twinx()
color = 'red'
ax2.set_ylabel("Recombination Rate (cm$^{-3}$s$^{-1}$)", color=color, fontsize=14)
p3, = ax2.plot(x_pos_cm, usrh, color=color, label="USRH")
ax2.tick_params(axis='y', labelcolor=color)
ax2.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

# Create a single combined legend for clarity.
plt.legend(handles=[p3, p1, p2], loc='best', fontsize=12)

fig.tight_layout()
plt.savefig("diode_recombination_currents.png")
print("Saved: diode_recombination_currents.png")

print("\n--- Plotting complete ---")