"""
diode_deepxde_pytorch.py - Stabilized 2D P-N Photodiode PINN Simulation
"""

import deepxde as dde
import torch
import numpy as np
import matplotlib.pyplot as plt
import time

# Set backend
try:
    dde.config.set_default_backend("pytorch")
except:
    print("Warning: Could not set backend explicitly")

# Device setup
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using: MPS (Apple Silicon)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using: CUDA")
else:
    device = torch.device("cpu")
    print("Using: CPU")

dde.config.set_default_float("float32")

# ======================================================================
# PHYSICAL CONSTANTS
# ======================================================================
q = 1.602e-19
k_B = 1.381e-23
eps_0 = 8.854e-14
T = 300.0
V_t = k_B * T / q  # ~0.0259 V

eps_r = 11.7
epsilon = eps_r * eps_0
n_i = 1.0e10
n_i_sq = n_i ** 2

mu_n = 1400.0
mu_p = 450.0
D_n = mu_n * V_t
D_p = mu_p * V_t

tau_n = 1.0e-6
tau_p = 1.0e-6

# ======================================================================
# GEOMETRY
# ======================================================================
width_cm = 100.0 * 1e-4
n_plus_thickness = 0.5 * 1e-4
p_thickness = 50.0 * 1e-4
p_plus_thickness = 10.0 * 1e-4

y_j1_cm = n_plus_thickness
y_j2_cm = n_plus_thickness + p_thickness
y_total_cm = n_plus_thickness + p_thickness + p_plus_thickness

N_D_nplus = 1e19
N_A_p = 1e16
N_A_pplus = 1e19

print(f"\nDevice: {width_cm * 1e4:.1f}µm × {y_total_cm * 1e4:.1f}µm")
print(f"n+: {n_plus_thickness * 1e4:.2f}µm, p: {p_thickness * 1e4:.1f}µm, p+: {p_plus_thickness * 1e4:.1f}µm\n")

geom = dde.geometry.Rectangle([0, 0], [width_cm, y_total_cm])


def doping_profile(x_in):
    """Net doping N_D - N_A"""
    y = x_in[:, 1:2]
    if not isinstance(y, torch.Tensor):
        y = torch.tensor(y, dtype=torch.float32)

    N_D = torch.where(y <= y_j1_cm,
                      torch.full_like(y, N_D_nplus),
                      torch.full_like(y, 0.0))

    N_A_val = torch.where(y <= y_j2_cm,
                          torch.full_like(y, N_A_p),
                          torch.full_like(y, N_A_pplus))
    N_A = torch.where(y > y_j1_cm, N_A_val, torch.full_like(y, 0.0))

    return N_D - N_A


# ======================================================================
# NORMALIZATION SCALES (Critical for stability!)
# ======================================================================
# We need to scale down the huge current/flux terms
# Typical current density scale J ~ q * mu * n * E ~ 1e-19 * 1000 * 1e19 * 1e5 = 1e8 A/cm²
scale_current = q * mu_n * N_D_nplus * 1e5  # ~1e8 A/cm²
scale_recomb = N_D_nplus / tau_n  # ~1e25 cm⁻³/s


# ======================================================================
# PDE SYSTEM - STABILIZED
# ======================================================================
def pde_system(x, u):
    """2D drift-diffusion equations with normalization"""
    psi = u[:, 0:1]
    log_n = u[:, 1:2]
    log_p = u[:, 2:3]

    # Clamp log values to prevent overflow
    log_n = torch.clamp(log_n, min=np.log(1e5), max=np.log(1e20))
    log_p = torch.clamp(log_p, min=np.log(1e5), max=np.log(1e20))

    n = torch.exp(log_n)
    p = torch.exp(log_p)

    # POISSON: Compute Laplacian manually
    dpsi_x = dde.grad.jacobian(psi, x, i=0, j=0)
    dpsi_y = dde.grad.jacobian(psi, x, i=0, j=1)
    d2psi_xx = dde.grad.jacobian(dpsi_x, x, i=0, j=0)
    d2psi_yy = dde.grad.jacobian(dpsi_y, x, i=0, j=1)
    laplacian_psi = d2psi_xx + d2psi_yy

    N_net = doping_profile(x)
    rho = q * (p - n + N_net)

    # Normalize Poisson equation (scale by characteristic charge density)
    norm_poisson = 1.0 / (q * N_A_pplus)
    eq1_poisson = (epsilon * laplacian_psi + rho) * norm_poisson

    # GRADIENTS
    dn_dx = dde.grad.jacobian(n, x, i=0, j=0)
    dn_dy = dde.grad.jacobian(n, x, i=0, j=1)
    dp_dx = dde.grad.jacobian(p, x, i=0, j=0)
    dp_dy = dde.grad.jacobian(p, x, i=0, j=1)

    # RECOMBINATION (stabilized)
    U_num = n * p - n_i_sq
    U_den = tau_p * (n + n_i) + tau_n * (p + n_i)
    U = U_num / (U_den + 1e-10)

    # ELECTRON CONTINUITY
    Jn_x = -q * mu_n * n * dpsi_x + q * D_n * dn_dx
    Jn_y = -q * mu_n * n * dpsi_y + q * D_n * dn_dy
    div_Jn = dde.grad.jacobian(Jn_x, x, i=0, j=0) + \
             dde.grad.jacobian(Jn_y, x, i=0, j=1)

    # Normalize by characteristic recombination rate
    eq2_electron = ((div_Jn / q) - U) / scale_recomb

    # HOLE CONTINUITY
    Jp_x = -q * mu_p * p * dpsi_x - q * D_p * dp_dx
    Jp_y = -q * mu_p * p * dpsi_y - q * D_p * dp_dy
    div_Jp = dde.grad.jacobian(Jp_x, x, i=0, j=0) + \
             dde.grad.jacobian(Jp_y, x, i=0, j=1)

    eq3_hole = (-(div_Jp / q) - U) / scale_recomb

    return [eq1_poisson, eq2_electron, eq3_hole]


# ======================================================================
# BOUNDARY CONDITIONS
# ======================================================================
V_bias = 0.0

# Cathode (n+ @ y=0)
V_bi_cathode = V_t * np.log(N_D_nplus / n_i)
n_eq_cathode = N_D_nplus
p_eq_cathode = n_i_sq / N_D_nplus
log_n_eq_cathode = np.log(n_eq_cathode)
log_p_eq_cathode = np.log(p_eq_cathode)
psi_cathode = V_bi_cathode + V_bias

# Anode (p+ @ y=y_total)
V_bi_anode = -V_t * np.log(N_A_pplus / n_i)
n_eq_anode = n_i_sq / N_A_pplus
p_eq_anode = N_A_pplus
log_n_eq_anode = np.log(n_eq_anode)
log_p_eq_anode = np.log(p_eq_anode)
psi_anode = V_bi_anode

print(f"Built-in voltage: {V_bi_cathode - V_bi_anode:.4f} V")
print(f"Cathode: ψ={psi_cathode:.3f}V, n={n_eq_cathode:.1e}, p={p_eq_cathode:.1e}")
print(f"Anode:   ψ={psi_anode:.3f}V, n={n_eq_anode:.1e}, p={p_eq_anode:.1e}\n")

# Contact regions
cathode_x_start = 45.0 * 1e-4
cathode_x_end = 55.0 * 1e-4


def on_cathode(x, on_boundary):
    is_on_y = np.isclose(x[1], 0, atol=1e-8)
    is_on_x = (x[0] >= cathode_x_start) & (x[0] <= cathode_x_end)
    return on_boundary and is_on_y and is_on_x


def on_anode(x, on_boundary):
    return on_boundary and np.isclose(x[1], y_total_cm, atol=1e-8)


# BCs
bc_psi_cathode = dde.icbc.DirichletBC(geom, lambda x: psi_cathode, on_cathode, component=0)
bc_logn_cathode = dde.icbc.DirichletBC(geom, lambda x: log_n_eq_cathode, on_cathode, component=1)
bc_logp_cathode = dde.icbc.DirichletBC(geom, lambda x: log_p_eq_cathode, on_cathode, component=2)

bc_psi_anode = dde.icbc.DirichletBC(geom, lambda x: psi_anode, on_anode, component=0)
bc_logn_anode = dde.icbc.DirichletBC(geom, lambda x: log_n_eq_anode, on_anode, component=1)
bc_logp_anode = dde.icbc.DirichletBC(geom, lambda x: log_p_eq_anode, on_anode, component=2)

bcs = [bc_psi_cathode, bc_logn_cathode, bc_logp_cathode,
       bc_psi_anode, bc_logn_anode, bc_logp_anode]

# ======================================================================
# MODEL SETUP
# ======================================================================
data = dde.data.PDE(
    geom,
    pde_system,
    bcs,
    num_domain=3000,  # Reduced for stability
    num_boundary=800,
    num_test=500,
    anchors=np.array([
        [width_cm / 2, y_j1_cm],
        [width_cm / 2, y_j2_cm],
    ])
)

# Smaller network for stability
layer_sizes = [2] + [64] * 4 + [3]
net = dde.nn.FNN(layer_sizes, "tanh", "Glorot normal")


# Better output transformation
def output_transform(x, u):
    """Transform to physical range with smooth scaling"""
    psi_out = u[:, 0:1] * 0.6  # Scale potential to ~[-0.6, 0.6]V range

    # Scale log concentrations smoothly
    log_center = (log_n_eq_cathode + log_n_eq_anode) / 2.0
    log_range = (log_n_eq_cathode - log_n_eq_anode) / 2.0

    log_n_out = log_center + log_range * torch.tanh(u[:, 1:2])
    log_p_out = log_center + log_range * torch.tanh(u[:, 2:3])

    return torch.cat((psi_out, log_n_out, log_p_out), dim=1)


net.apply_output_transform(output_transform)
model = dde.Model(data, net)

# ======================================================================
# TRAINING - GRADUAL APPROACH
# ======================================================================
print("=" * 60)
print("Training...")
print("=" * 60)

# Start with much higher BC weights to enforce boundaries first
loss_weights = [1.0, 1.0, 1.0] + [1000.0] * len(bcs)

# Phase 1: ADAM with low learning rate
print("\nPhase 1: ADAM (warm-up)")
start = time.time()
model.compile("adam", lr=1e-4, loss_weights=loss_weights)
losshistory, train_state = model.train(iterations=5000, display_every=1000)
print(f"✓ ADAM warm-up: {time.time() - start:.1f}s")

# Phase 2: ADAM with standard learning rate
print("\nPhase 2: ADAM (main)")
start = time.time()
model.compile("adam", lr=5e-4, loss_weights=loss_weights)
losshistory, train_state = model.train(iterations=10000, display_every=2000)
print(f"✓ ADAM main: {time.time() - start:.1f}s")

# Phase 3: L-BFGS (fine-tuning)
print("\nPhase 3: L-BFGS (refinement)")
start = time.time()
# Reduce BC weights slightly for L-BFGS
loss_weights = [1.0, 1.0, 1.0] + [50.0] * len(bcs)
model.compile("L-BFGS", loss_weights=loss_weights)
losshistory, train_state = model.train(display_every=500)
print(f"✓ L-BFGS: {time.time() - start:.1f}s")

print(f"\nFinal losses - Train: {train_state.loss_train[-1]:.2e}, Test: {train_state.loss_test[-1]:.2e}")

# ======================================================================
# VISUALIZATION
# ======================================================================
print("\n" + "=" * 60)
print("Generating plots...")
print("=" * 60)

# 1D Profile
n_points = 500
y_plot = np.linspace(0, y_total_cm, n_points)
x_plot = np.full_like(y_plot, width_cm / 2.0)
plot_domain = np.vstack((x_plot, y_plot)).T

u_pred = model.predict(plot_domain)
psi_pred = u_pred[:, 0]
n_pred = np.exp(np.clip(u_pred[:, 1], np.log(1e5), np.log(1e20)))
p_pred = np.exp(np.clip(u_pred[:, 2], np.log(1e5), np.log(1e20)))
y_um = y_plot * 1e4

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle(f'2D Photodiode PINN (V={V_bias}V)', fontsize=14, fontweight='bold')

# Potential
ax1.plot(y_um, psi_pred, 'r-', lw=2.5)
ax1.axhline(0, color='k', linestyle=':', alpha=0.3)
ax1.axvline(y_j1_cm * 1e4, color='b', linestyle='--', alpha=0.5, label='n⁺/p')
ax1.axvline(y_j2_cm * 1e4, color='g', linestyle='--', alpha=0.5, label='p/p⁺')
ax1.set_xlabel('Depth (µm)', fontsize=12)
ax1.set_ylabel('Potential ψ (V)', fontsize=12)
ax1.set_title('Electrostatic Potential')
ax1.grid(True, alpha=0.3)
ax1.legend()

# Carriers
ax2.semilogy(y_um, n_pred, 'b-', lw=2.5, label='Electrons (n)')
ax2.semilogy(y_um, p_pred, 'r-', lw=2.5, label='Holes (p)')
ax2.semilogy(y_um, np.full_like(y_um, n_i), 'k:', lw=2, label='$n_i$')
ax2.axvline(y_j1_cm * 1e4, color='gray', linestyle='--', alpha=0.3)
ax2.axvline(y_j2_cm * 1e4, color='gray', linestyle='--', alpha=0.3)
ax2.set_xlabel('Depth (µm)', fontsize=12)
ax2.set_ylabel('Concentration (cm⁻³)', fontsize=12)
ax2.set_title('Carrier Concentrations')
ax2.legend()
ax2.grid(True, alpha=0.3, which='both')
ax2.set_ylim(1e6, 1e20)

plt.tight_layout()
plt.savefig("photodiode_results.png", dpi=150, bbox_inches='tight')
print("✓ Saved: photodiode_results.png")

# Export data
np.savez_compressed(
    "photodiode_data.npz",
    y=y_plot, psi=psi_pred, n=n_pred, p=p_pred,
    width_cm=width_cm, y_total_cm=y_total_cm,
    y_j1_cm=y_j1_cm, y_j2_cm=y_j2_cm, V_bias=V_bias
)
print("✓ Saved: photodiode_data.npz")

print("\n" + "=" * 60)
print("Summary:")
print(f"  Built-in voltage: {V_bi_cathode - V_bi_anode:.4f} V")
print(f"  Final train loss: {train_state.loss_train[-1]:.2e}")
print(f"  Final test loss:  {train_state.loss_test[-1]:.2e}")
print("=" * 60)
print("\n✅ Complete!\n")