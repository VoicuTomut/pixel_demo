"""
diode_deepxde_v7_scaled.py

Final attempt at a robust 2D photodiode simulator.
This version implements full non-dimensionalization to stabilize
the equations, which is a standard technique for this class of
numerically "stiff" problems.

All variables are scaled to be O(1).
- Potentials (psi, phin, phip) are scaled by V_t.
- Lengths (x, y) are scaled by the device thickness L.
- Doping (N) is scaled by N_max.
- Carriers (n, p) are scaled by N_max.

This completely rewrites the PDE system into a stable, unit-less form.
"""

import deepxde as dde
import torch
import numpy as np
import matplotlib.pyplot as plt
import time
import scipy.integrate as spi

# Set backend
try:
    dde.config.set_default_backend("pytorch")
    print("Using: PyTorch")
except:
    print("Warning: Could not set backend explicitly")

dde.config.set_default_float("float32")

# ======================================================================
# 1. PHYSICAL CONSTANTS
# ======================================================================
q = 1.602e-19
k_B = 1.381e-23
eps_0 = 8.854e-14
T = 300.0
V_t = k_B * T / q  # ~0.0259 V (This is our scaling voltage, V_sc)

eps_r = 11.7
epsilon = eps_r * eps_0
n_i_phys = 1.0e10  # Physical n_i

mu_n_phys = 1400.0
mu_p_phys = 450.0

tau_n_phys = 1.0e-6
tau_p_phys = 1.0e-6

# ======================================================================
# 2. SCALING & GEOMETRY
# ======================================================================

# --- Physical Dimensions ---
width_phys = 100.0 * 1e-4
n_plus_thickness_phys = 1.0 * 1e-4
p_thickness_phys = 30.0 * 1e-4
p_plus_thickness_phys = 5.0 * 1e-4
L_sc = n_plus_thickness_phys + p_thickness_phys + p_plus_thickness_phys  # 36e-4 cm

# --- Scaled (Unit-less) Dimensions ---
# All lengths are now divided by L_sc
width_sc = width_phys / L_sc
y_j1_sc = n_plus_thickness_phys / L_sc
y_j2_sc = (n_plus_thickness_phys + p_thickness_phys) / L_sc
y_total_sc = 1.0  # y_total_phys / L_sc

# --- Scaled (Unit-less) Doping ---
N_D_nplus_phys = 1e18
N_A_p_phys = 1e15
N_A_pplus_phys = 1e18
N_sc = max(N_D_nplus_phys, N_A_pplus_phys)  # 1e18 (This is our scaling density)

N_D_nplus_sc = N_D_nplus_phys / N_sc
N_A_p_sc = N_A_p_phys / N_sc
N_A_pplus_sc = N_A_pplus_phys / N_sc
n_i_sc = n_i_phys / N_sc  # ~1e-8
n_i_sq_sc = n_i_sc ** 2

# --- Scaled (Unit-less) Physics Params ---
mu_n_sc = mu_n_phys * V_t / (L_sc ** 2)  # D = mu * V_t. This is D / L^2
mu_p_sc = mu_p_phys * V_t / (L_sc ** 2)  # D / L^2
# Note: tau_n and tau_p are already in seconds, no scaling needed

# Poisson prefactor: (eps * V_t) / (q * N_sc * L_sc^2)
poisson_prefactor_sc = (epsilon * V_t) / (q * N_sc * (L_sc ** 2))

# R_thermal prefactor: (L_sc^2) / D_n
# We'll scale continuity by R_thermal to start
R_thermal_sc = (n_i_sq_sc) / (tau_p_phys * (n_i_sc) + tau_n_phys * (n_i_sc))
# G_sc = R_thermal_sc (for dark)
continuity_scaler_sc = R_thermal_sc

print("--- SCALED PARAMETERS ---")
print(f"L_sc (length): {L_sc:.2e} cm")
print(f"N_sc (density): {N_sc:.2e} cm⁻³")
print(f"V_sc (potential): {V_t:.3f} V")
print(f"n_i_sc (scaled n_i): {n_i_sc:.2e}")
print(f"Poisson Prefactor: {poisson_prefactor_sc:.2e}")
print(f"R_thermal_sc: {R_thermal_sc:.2e}")
print(f"Scaled Geometry: y_j1={y_j1_sc:.3f}, y_j2={y_j2_sc:.3f}, width={width_sc:.2f}")

# --- Scaled (Unit-less) Geometry ---
geom = dde.geometry.Rectangle([0, 0], [width_sc, y_total_sc])


def doping_profile_sc(x_in):
    """Net doping (N_D - N_A) in SCALED units"""
    y_sc = x_in[:, 1:2]
    if not isinstance(y_sc, torch.Tensor):
        y_sc = torch.tensor(y_sc, dtype=torch.float32)

    N_D_sc = torch.where(y_sc <= y_j1_sc,
                         torch.full_like(y_sc, N_D_nplus_sc),
                         torch.full_like(y_sc, 0.0))

    N_A_val_sc = torch.where(y_sc <= y_j2_sc,
                             torch.full_like(y_sc, N_A_p_sc),
                             torch.full_like(y_sc, N_A_pplus_sc))
    N_A_sc = torch.where(y_sc > y_j1_sc, N_A_val_sc, torch.full_like(y_sc, 0.0))

    return N_D_sc - N_A_sc


# ======================================================================
# 3. OPTICAL GENERATION (SCALED)
# ======================================================================

class OpticalGeneration:
    def __init__(self):
        self.G0_sc = 0.0
        self.alpha_sc = 0.0
        self.is_dark = True
        self.R_thermal_sc = R_thermal_sc

    def set_dark(self):
        self.is_dark = True
        self.G0_sc = 0.0
        self.alpha_sc = 0.0

    def set_light(self, lambda_nm, P_opt_W_cm2):
        # Physical alpha
        if lambda_nm > 1100:
            alpha_phys = 10.0
        elif lambda_nm > 1000:
            alpha_phys = 100.0
        elif lambda_nm > 900:
            alpha_phys = 800.0
        elif lambda_nm > 800:
            alpha_phys = 3000.0
        elif lambda_nm > 700:
            alpha_phys = 8000.0
        elif lambda_nm > 600:
            alpha_phys = 10000.0
        else:
            alpha_phys = 50000.0

        # Scaled alpha (alpha' = alpha * L)
        self.alpha_sc = alpha_phys * L_sc

        R_surface = 0.3
        h_J = 6.626e-34
        c_m_s = 2.998e8
        lambda_m = lambda_nm * 1e-9
        E_photon_J = h_J * c_m_s / lambda_m
        Phi_0 = P_opt_W_cm2 / E_photon_J

        G0_phys = Phi_0 * (1.0 - R_surface) * alpha_phys

        # Scale G0: G' = G / N_sc
        self.G0_sc = G0_phys / N_sc
        self.is_dark = False
        print(f"  [Gen] λ={lambda_nm:.0f}nm, α_sc={self.alpha_sc:.2f}, G₀_sc={self.G0_sc:.2e}")

    def get_generation_term_sc(self, y_sc):
        if self.is_dark:
            return torch.zeros_like(y_sc)
        else:
            return self.G0_sc * torch.exp(-self.alpha_sc * y_sc)

    def get_continuity_scaler_sc(self):
        # Scaler is G' or R'
        return max(self.G0_sc, self.R_thermal_sc, 1e-10)


optical_gen = OpticalGeneration()


# ======================================================================
# 4. PDE SYSTEM (SCALED QFP)
# ======================================================================

def pde_system_scaled(x, u):
    """
    Solves the 2D drift-diffusion system using SCALED QFP variables.
    x = [x_sc, y_sc]
    u = [psi_sc, phi_n_sc, phi_p_sc]
    """
    psi_sc = u[:, 0:1]
    phi_n_sc = u[:, 1:2]
    phi_p_sc = u[:, 2:3]

    # --- QFP Formulation (Scaled) ---
    exp_arg_n = torch.clamp(psi_sc - phi_n_sc, -80, 80)
    exp_arg_p = torch.clamp(phi_p_sc - psi_sc, -80, 80)

    n_sc = torch.clamp(n_i_sc * torch.exp(exp_arg_n), max=1.5)
    p_sc = torch.clamp(n_i_sc * torch.exp(exp_arg_p), max=1.5)

    # --- 1. Poisson's Equation ---
    dpsi_x = dde.grad.jacobian(psi_sc, x, i=0, j=0)
    dpsi_y = dde.grad.jacobian(psi_sc, x, i=0, j=1)
    laplacian_psi = dde.grad.jacobian(dpsi_x, x, i=0, j=0) + \
                    dde.grad.jacobian(dpsi_y, x, i=0, j=1)

    N_net_sc = doping_profile_sc(x)
    eq1_poisson = poisson_prefactor_sc * laplacian_psi + (p_sc - n_sc + N_net_sc)

    # --- 2. Recombination & Generation ---
    U_num_sc = n_sc * p_sc - n_i_sq_sc
    U_den_sc = tau_p_phys * (n_sc + n_i_sc) + tau_n_phys * (p_sc + n_i_sc)
    U_sc = U_num_sc / (U_den_sc + 1e-20)  # U' = U / N_sc

    G_sc = optical_gen.get_generation_term_sc(x[:, 1:2])  # G' = G / N_sc

    S_cont_sc = optical_gen.get_continuity_scaler_sc()
    RG_term = (G_sc - U_sc) / S_cont_sc

    # --- 3. Electron Continuity Equation ---
    dphin_x = dde.grad.jacobian(phi_n_sc, x, i=0, j=0)
    dphin_y = dde.grad.jacobian(phi_n_sc, x, i=0, j=1)

    # D = mu * V_t. J_n' = (D_n / L^2) * n_sc * grad(phi_n_sc)
    n_floor = n_sc + (n_i_sc * 1e-3)  # Add small floor
    term_n_x = mu_n_sc * n_floor * dphin_x
    term_n_y = mu_n_sc * n_floor * dphin_y

    div_Jn_term = dde.grad.jacobian(term_n_x, x, i=0, j=0) + \
                  dde.grad.jacobian(term_n_y, x, i=0, j=1)

    eq2_electron = div_Jn_term / S_cont_sc - RG_term

    # --- 4. Hole Continuity Equation ---
    dphip_x = dde.grad.jacobian(phi_p_sc, x, i=0, j=0)
    dphip_y = dde.grad.jacobian(phi_p_sc, x, i=0, j=1)

    p_floor = p_sc + (n_i_sc * 1e-3)  # Add small floor
    term_p_x = mu_p_sc * p_floor * dphip_x
    term_p_y = mu_p_sc * p_floor * dphip_y

    div_Jp_term = dde.grad.jacobian(term_p_x, x, i=0, j=0) + \
                  dde.grad.jacobian(term_p_y, x, i=0, j=1)

    eq3_hole = div_Jp_term / S_cont_sc + RG_term

    return [eq1_poisson, eq2_electron, eq3_hole]


# ======================================================================
# 5. BOUNDARY CONDITIONS (SCALED)
# ======================================================================
cathode_x_start_sc = (40.0 * 1e-4) / L_sc
cathode_x_end_sc = (60.0 * 1e-4) / L_sc

# V_bi_cathode_sc = V_bi_cathode_phys / V_t
V_bi_cathode_sc = np.log(N_D_nplus_phys / n_i_phys)
V_bi_anode_sc = -np.log(N_A_pplus_phys / n_i_phys)

print(f"Scaled Built-in voltage: {V_bi_cathode_sc - V_bi_anode_sc:.4f} (in V_t)")


def on_cathode(x, on_boundary):
    is_on_y = np.isclose(x[1], 0, atol=1e-8)
    is_on_x = (x[0] >= cathode_x_start_sc) & (x[0] <= cathode_x_end_sc)
    return on_boundary and is_on_y and is_on_x


def on_anode(x, on_boundary):
    return on_boundary and np.isclose(x[1], y_total_sc, atol=1e-8)


def get_bcs_scaled(V_bias_phys=0.0):
    V_bias_sc = V_bias_phys / V_t  # Scale the bias voltage

    psi_cathode_val = V_bi_cathode_sc + V_bias_sc
    phin_cathode_val = V_bias_sc
    phip_cathode_val = V_bias_sc

    bc_psi_cathode = dde.icbc.DirichletBC(geom, lambda x: psi_cathode_val, on_cathode, component=0)
    bc_phin_cathode = dde.icbc.DirichletBC(geom, lambda x: phin_cathode_val, on_cathode, component=1)
    bc_phip_cathode = dde.icbc.DirichletBC(geom, lambda x: phip_cathode_val, on_cathode, component=2)

    psi_anode_val = V_bi_anode_sc  # V_bias = 0
    phin_anode_val = 0.0
    phip_anode_val = 0.0

    bc_psi_anode = dde.icbc.DirichletBC(geom, lambda x: psi_anode_val, on_anode, component=0)
    bc_phin_anode = dde.icbc.DirichletBC(geom, lambda x: phin_anode_val, on_anode, component=1)
    bc_phip_anode = dde.icbc.DirichletBC(geom, lambda x: phin_anode_val, on_anode, component=2)

    return [bc_psi_cathode, bc_phin_cathode, bc_phip_cathode,
            bc_psi_anode, bc_phin_anode, bc_phip_anode]


# ======================================================================
# 6. CURRENT CALCULATION (SCALED)
# ======================================================================

def J_y_total_scaled(x, u):
    """Calculate total vertical current density from SCALED variables"""
    psi_sc = u[:, 0:1]
    phi_n_sc = u[:, 1:2]
    phi_p_sc = u[:, 2:3]

    exp_arg_n = torch.clamp(psi_sc - phi_n_sc, -80, 80)
    exp_arg_p = torch.clamp(phi_p_sc - psi_sc, -80, 80)

    n_sc = torch.clamp(n_i_sc * torch.exp(exp_arg_n), max=1.5)
    p_sc = torch.clamp(n_i_sc * torch.exp(exp_arg_p), max=1.5)

    n_floor = n_sc + (n_i_sc * 1e-3)
    p_floor = p_sc + (n_i_sc * 1e-3)

    dphin_y = dde.grad.jacobian(phi_n_sc, x, i=0, j=1)
    dphip_y = dde.grad.jacobian(phi_p_sc, x, i=0, j=1)

    # J_n_y_sc = (D_n / L^2) * n_sc * grad(phi_n_sc)
    Jn_y_sc = -mu_n_sc * n_floor * dphin_y
    Jp_y_sc = -mu_p_sc * p_floor * dphip_y

    return Jn_y_sc + Jp_y_sc


def calculate_current_scaled(model, n_points_x=50):
    """
    Calculate terminal current from SCALED model.
    """
    x_contact_sc = np.linspace(0, width_sc, n_points_x)
    y_contact_sc = np.full_like(x_contact_sc, y_total_sc)
    contact_points_sc = np.vstack((x_contact_sc, y_contact_sc)).T

    J_y_values_sc = model.predict(contact_points_sc, operator=J_y_total_scaled)
    J_y_values_sc = J_y_values_sc.flatten()

    # Integrate J' over x'
    I_sc = -spi.trapezoid(J_y_values_sc, x_contact_sc)

    # Convert scaled current I' back to physical current I
    # J_phys = (q * D * N_sc / L_sc) * J'
    # I_phys = J_phys * (L_sc * width_sc) = (q * D * N_sc) * width_sc * I'
    # I_phys_per_cm = I_phys / (width_sc * L_sc) = (q * D * N_sc / L_sc) * I'

    # J_phys = (q * mu * V_t * N_sc / L_sc) * J'
    # I' = integral(J' dx') = integral(J' * d(x/L))
    # I_phys = integral(J_phys dx) = integral(J_phys * d(x' * L)) = L * integral(J_phys dx')
    # J_phys = (q * D * N_sc / L_sc) * J_sc'
    # Current scaling factor: J_0 = q * D_n * N_sc / L_sc
    # We use D_n as the reference diffusivity
    J_0 = q * (mu_n_phys * V_t) * N_sc / L_sc

    # I' = integral(J' dx')
    # I_phys = integral(J_phys dx) = integral( (J_0 * J') * (L_sc * dx') )
    # I_phys = J_0 * L_sc * integral(J' dx') = J_0 * L_sc * I'
    I_phys = (J_0 * L_sc) * I_sc

    # Current per cm (width)
    I_phys_per_cm = I_phys / (width_sc * L_sc)

    if np.isnan(I_phys_per_cm):
        print("  WARNING: Current calculation resulted in NaN. Returning 0.")
        return 0.0

    return I_phys_per_cm


# ======================================================================
# 7. SOLVER
# ======================================================================
layer_sizes = [2] + [80] * 4 + [3]
net = dde.nn.FNN(layer_sizes, "tanh", "Glorot normal")
weights_dark_0V = None


def solve_photodiode_scaled(V_bias_phys, iters_stage1, iters_stage2, iters_stage3, initial_weights=None):
    global net

    gen_status = "DARK" if optical_gen.is_dark else "LIGHT"
    print("-" * 60)
    print(f"Solving for V_bias = {V_bias_phys:.3f} V, {gen_status}")
    print(f"  Continuity Scaler: {optical_gen.get_continuity_scaler_sc():.2e}")

    bcs = get_bcs_scaled(V_bias_phys)
    data = dde.data.PDE(
        geom,
        pde_system_scaled,
        bcs,
        num_domain=3000,
        num_boundary=1200,
        num_test=1000
    )

    model = dde.Model(data, net)

    if initial_weights:
        print("  Using transfer learning...")
        model.net.load_state_dict(initial_weights)
    else:
        print("  Initializing network from scratch...")

        def init_weights(m):
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_normal_(m.weight)
                m.bias.data.fill_(0.0)

        net.apply(init_weights)

    loss_weights = [1.0] * 3 + [500.0] * len(bcs)
    print(f"  Using manual loss weights: [PDEs: 1.0], [BCs: 500.0]")

    # --- STAGE 1: High LR ---
    if iters_stage1 > 0:
        print(f"\n  --- STAGE 1 (ADAM, lr=1e-3, {iters_stage1} iters) ---")
        start = time.time()
        model.compile("adam", lr=1e-3, loss_weights=loss_weights)
        losshistory, train_state = model.train(
            iterations=iters_stage1,
            display_every=max(1, iters_stage1 // 10)
        )
        print(f"  ✓ STAGE 1: {time.time() - start:.1f}s, Final loss: {train_state.best_loss_train:.2e}")

    # --- STAGE 2: Medium LR ---
    if iters_stage2 > 0:
        print(f"\n  --- STAGE 2 (ADAM, lr=1e-4, {iters_stage2} iters) ---")
        start = time.time()
        model.compile("adam", lr=1e-4, loss_weights=loss_weights)
        losshistory, train_state = model.train(
            iterations=iters_stage2,
            display_every=max(1, iters_stage2 // 10)
        )
        print(f"  ✓ STAGE 2: {time.time() - start:.1f}s, Final loss: {train_state.best_loss_train:.2e}")

    # --- STAGE 3: Low LR ---
    if iters_stage3 > 0:
        print(f"\n  --- STAGE 3 (ADAM, lr=1e-5, {iters_stage3} iters) ---")
        start = time.time()
        model.compile("adam", lr=1e-5, loss_weights=loss_weights)
        losshistory, train_state = model.train(
            iterations=iters_stage3,
            display_every=max(1, iters_stage3 // 10)
        )
        print(f"  ✓ STAGE 3: {time.time() - start:.1f}s, Final loss: {train_state.best_loss_train:.2e}")

    import copy
    return model, copy.deepcopy(model.net.state_dict())


# ======================================================================
# 8. MAIN EXECUTION
# ======================================================================

if __name__ == "__main__":


    V_sweep = np.array([-0.5, 0.0, 0.5])

    iters_s1_first = 20000
    iters_s2_first = 10000
    iters_s3_first = 5000

    iters_s1_transfer = 8000
    iters_s2_transfer = 4000
    iters_s3_transfer = 2000

    # --- CURRICULUM: Solve V=0.0 DARK first ---
    print("\n" + "=" * 70)
    print("PRE-TRAINING: Solving V=0.0 DARK for stable initial guess")
    print("=" * 70)
    optical_gen.set_dark()

    model_dark_0V, weights_dark_0V = solve_photodiode_scaled(
        0.0, iters_s1_first, iters_s2_first, iters_s3_first, initial_weights=None
    )
    I_dark_0V_calc = calculate_current_scaled(model_dark_0V)
    print(f"  >>> I_dark(0.000V) = {I_dark_0V_calc:.3e} A/cm")

    V_sweep_dark = [0.0]
    currents_dark_vals = [I_dark_0V_calc]

    # --- DARK I-V SWEEP ---
    print("\n" + "=" * 70)
    print("SCALED DARK I-V SWEEP")
    print("=" * 70)
    optical_gen.set_dark()
    for V in V_sweep:
        if V == 0.0: continue
        V_sweep_dark.append(V)
        model, _ = solve_photodiode_scaled(
            V, iters_s1_transfer, iters_s2_transfer, iters_s3_transfer,
            initial_weights=weights_dark_0V
        )
        current = calculate_current_scaled(model)
        print(f"  >>> I_dark({V:.3f}V) = {current:.3e} A/cm")
        currents_dark_vals.append(current)

    dark_results = sorted(zip(V_sweep_dark, currents_dark_vals))
    V_sweep_dark_sorted = [v for v, i in dark_results]
    currents_dark_sorted = [i for v, i in dark_results]

    # --- LIGHT I-V SWEEP ---
    print("\n" + "=" * 70)
    print("SCALED LIGHT I-V SWEEP (850nm, 1 mW/cm²)")
    print("=" * 70)
    optical_gen.set_light(850, P_opt_W_cm2=0.001)

    V_sweep_light = []
    currents_light_vals = []
    for V in V_sweep:
        V_sweep_light.append(V)
        model, _ = solve_photodiode_scaled(
            V, iters_s1_transfer, iters_s2_transfer, iters_s3_transfer,
            initial_weights=weights_dark_0V
        )
        current = calculate_current_scaled(model)
        print(f"  >>> I_light({V:.3f}V) = {current:.3e} A/cm")
        currents_light_vals.append(current)

    light_results = sorted(zip(V_sweep_light, currents_light_vals))
    V_sweep_light_sorted = [v for v, i in light_results]
    currents_light_sorted = [i for v, i in light_results]

    # --- PLOT I-V ---
    plt.figure(figsize=(10, 7))
    plt.plot(V_sweep_dark_sorted, np.array(currents_dark_sorted) * 1e3, 'bo-', label='Dark', linewidth=2)
    plt.plot(V_sweep_light_sorted, np.array(currents_light_sorted) * 1e3, 'ro-', label='Illuminated (850nm)',
             linewidth=2)
    plt.axhline(0, color='k', linestyle='--', alpha=0.3)
    plt.axvline(0, color='k', linestyle='--', alpha=0.3)
    plt.xlabel('Voltage (V)', fontsize=14)
    plt.ylabel('Current (mA/cm)', fontsize=14)
    plt.title('Photodiode I-V (Scaled QFP v7)', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig("photodiode_IV_scaled.png", dpi=150)
    print("✓ Saved: photodiode_IV_scaled.png")

    # --- QE SWEEP ---
    print("\n" + "=" * 70)
    print("SCALED QUANTUM EFFICIENCY SWEEP (V=0)")
    print("=" * 70)
    print(f"Using pre-calculated I_dark(0V) = {I_dark_0V_calc:.3e} A/cm")

    wavelengths_nm = np.array([400, 700, 950])
    photocurrents = []
    EQE_values = []
    P_opt = 0.001  # 1 mW/cm²

    for lam_nm in wavelengths_nm:
        optical_gen.set_light(lam_nm, P_opt_W_cm2=P_opt)
        model_light, _ = solve_photodiode_scaled(
            0.0, iters_s1_transfer, iters_s2_transfer, iters_s3_transfer,
            initial_weights=weights_dark_0V
        )
        I_light = calculate_current_scaled(model_light)
        I_photo = I_light - I_dark_0V_calc
        photocurrents.append(I_photo)

        # Calculate EQE (using all physical values)
        h_J = 6.626e-34
        c_m_s = 2.998e8
        lambda_m = lam_nm * 1e-9
        E_photon_J = h_J * c_m_s / lambda_m
        Phi_0_phys = P_opt / E_photon_J

        photons_in_per_cm = Phi_0_phys * width_phys
        electrons_out_per_cm = abs(I_photo) / q

        EQE = (electrons_out_per_cm / photons_in_per_cm) if photons_in_per_cm > 0 else 0
        EQE_values.append(EQE * 100)
        print(f"  >>> λ={lam_nm}nm: I_photo={I_photo:.3e} A/cm, EQE={EQE * 100:.1f}%")

    # --- PLOT QE ---
    plt.figure(figsize=(10, 7))
    plt.plot(wavelengths_nm, EQE_values, 'go-', linewidth=2, markersize=8)
    plt.xlabel('Wavelength (nm)', fontsize=14)
    plt.ylabel('EQE (%)', fontsize=14)
    plt.title('Photodiode Spectral Response (Scaled QFP v7)', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 100)
    plt.tight_layout()
    plt.savefig("photodiode_QE_scaled.png", dpi=150)
    print("✓ Saved: photodiode_QE_scaled.png")

    print("\n✅ Scaled simulation run complete!")