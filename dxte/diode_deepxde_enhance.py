"""
diode_deepxde_robust_v6.py

A robust 2D photodiode simulator using the Quasi-Fermi Potential (QFP)
formulation and advanced stabilization techniques.

(Fix 6: Corrected SyntaxError: "name '...' is assigned to before
 global declaration" by moving the global declaration to the top
 of the __main__ block.)
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

# Device setup (optional, for MPS/CUDA)
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
# PHYSICAL CONSTANTS (Unchanged)
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

tau_n = 1.0e-6
tau_p = 1.0e-6

# ======================================================================
# GEOMETRY & DOPING (Unchanged, but with new characteristic scales)
# ======================================================================
width_cm = 100.0 * 1e-4
n_plus_thickness = 1.0 * 1e-4
p_thickness = 30.0 * 1e-4
p_plus_thickness = 5.0 * 1e-4

y_j1_cm = n_plus_thickness
y_j2_cm = n_plus_thickness + p_thickness
y_total_cm = n_plus_thickness + p_thickness + p_plus_thickness

N_D_nplus = 1e18
N_A_p = 1e15
N_A_pplus = 1e18

geom = dde.geometry.Rectangle([0, 0], [width_cm, y_total_cm])

# --- ROBUSTNESS: Define characteristic scales ---
N_max = max(N_D_nplus, N_A_pplus)
R_thermal = n_i / min(tau_n, tau_p)  # ~1e16
NUMERICAL_FLOOR = 1e4  # A small number to prevent 0*inf

print(f"Characteristic Scales:")
print(f"  N_max (Poisson): {N_max:.1e} cm⁻³")
print(f"  R_thermal (Continuity): {R_thermal:.1e} cm⁻³s⁻¹")


def doping_profile(x_in):
    """Net doping N_D - N_A. Unchanged."""
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
# OPTICAL GENERATION (Modified for robust scaling)
# ======================================================================

class OpticalGeneration:
    """Class to manage optical generation and scaling factors"""

    def __init__(self):
        self.G0 = 0.0
        self.alpha = 0.0
        self.is_dark = True
        self.R_thermal = R_thermal  # Store thermal rate

    def set_dark(self):
        self.is_dark = True
        self.G0 = 0.0
        self.alpha = 0.0

    def set_light(self, lambda_nm, P_opt_W_cm2):
        if lambda_nm > 1100:
            alpha_cm = 10.0
        elif lambda_nm > 1000:
            alpha_cm = 100.0
        elif lambda_nm > 900:
            alpha_cm = 800.0
        elif lambda_nm > 800:
            alpha_cm = 3000.0
        elif lambda_nm > 700:
            alpha_cm = 8000.0
        elif lambda_nm > 600:
            alpha_cm = 10000.0
        else:
            alpha_cm = 50000.0

        R_surface = 0.3
        h_J = 6.626e-34
        c_m_s = 2.998e8
        lambda_m = lambda_nm * 1e-9
        E_photon_J = h_J * c_m_s / lambda_m
        Phi_0 = P_opt_W_cm2 / E_photon_J

        self.G0 = Phi_0 * (1.0 - R_surface) * alpha_cm
        self.alpha = alpha_cm
        self.is_dark = False

        print(f"  [Gen] λ={lambda_nm:.0f}nm, α={alpha_cm:.0f} cm⁻¹, Φ₀={Phi_0:.2e} ph/s·cm², G₀={self.G0:.2e} cm⁻³s⁻¹")

    def get_generation_term(self, y):
        """Returns the generation rate G(y)"""
        if self.is_dark:
            return torch.zeros_like(y)
        else:
            return self.G0 * torch.exp(-self.alpha * y)

    def get_continuity_scaler(self):
        """
        ROBUSTNESS TRICK:
        Returns the scaling factor for the continuity equations.
        This ensures the R-G term is always O(1).
        """
        return max(self.G0, self.R_thermal, 1e10)  # Add 1e10 floor


# Create global instance
optical_gen = OpticalGeneration()


# ======================================================================
# PDE SYSTEM (QFP FORMULATION + ROBUST SCALING)
# ======================================================================

def pde_system_robust(x, u):
    """
    Solves the 2D drift-diffusion system using the
    Quasi-Fermi Potential (QFP) formulation for stability.

    Outputs u = [psi, phi_n, phi_p]
    """
    psi = u[:, 0:1]  # Electrostatic Potential (V)
    phi_n = u[:, 1:2]  # Electron QFP (V)
    phi_p = u[:, 2:3]  # Hole QFP (V)

    # --- QFP Formulation ---
    # STABILITY TRICK 1: Clamp arguments to exp() to avoid Inf/NaN
    exp_arg_n = torch.clamp((psi - phi_n) / V_t, -80, 80)
    exp_arg_p = torch.clamp((phi_p - psi) / V_t, -80, 80)

    n_raw = n_i * torch.exp(exp_arg_n)
    p_raw = n_i * torch.exp(exp_arg_p)

    # STABILITY TRICK 2 (FIXED): Clamp ONLY the max value
    # We must allow n and p to be very small (e.g. 1e2)
    n = torch.clamp(n_raw, max=N_max * 1.5)
    p = torch.clamp(p_raw, max=N_max * 1.5)

    # --- 1. Poisson's Equation ---
    dpsi_x = dde.grad.jacobian(psi, x, i=0, j=0)
    dpsi_y = dde.grad.jacobian(psi, x, i=0, j=1)
    d2psi_xx = dde.grad.jacobian(dpsi_x, x, i=0, j=0)
    d2psi_yy = dde.grad.jacobian(dpsi_y, x, i=0, j=1)
    laplacian_psi = d2psi_xx + d2psi_yy

    N_net = doping_profile(x)

    poisson_prefactor = epsilon / (q * N_max)
    eq1_poisson = poisson_prefactor * laplacian_psi + (p - n + N_net) / N_max

    # --- 2. Recombination & Generation ---
    U_num = n * p - n_i_sq
    # We add the floor here to prevent division by zero if n,p -> 0
    U_den = tau_p * (n + n_i + NUMERICAL_FLOOR) + tau_n * (p + n_i + NUMERICAL_FLOOR)
    U = U_num / U_den

    y = x[:, 1:2]
    G = optical_gen.get_generation_term(y)

    S_cont = optical_gen.get_continuity_scaler()
    RG_term = (G - U) / S_cont

    # --- 3. Electron Continuity Equation ---
    dphin_x = dde.grad.jacobian(phi_n, x, i=0, j=0)
    dphin_y = dde.grad.jacobian(phi_n, x, i=0, j=1)

    # STABILITY TRICK 3: Add numerical floor to n,p multipliers
    n_floor = n + NUMERICAL_FLOOR
    term_n_x = mu_n * n_floor * dphin_x
    term_n_y = mu_n * n_floor * dphin_y

    div_Jn_term = dde.grad.jacobian(term_n_x, x, i=0, j=0) + \
                  dde.grad.jacobian(term_n_y, x, i=0, j=1)

    eq2_electron = div_Jn_term / S_cont - RG_term

    # --- 4. Hole Continuity Equation ---
    dphip_x = dde.grad.jacobian(phi_p, x, i=0, j=0)
    dphip_y = dde.grad.jacobian(phi_p, x, i=0, j=1)

    # STABILITY TRICK 3: Add numerical floor to n,p multipliers
    p_floor = p + NUMERICAL_FLOOR
    term_p_x = mu_p * p_floor * dphip_x
    term_p_y = mu_p * p_floor * dphip_y

    div_Jp_term = dde.grad.jacobian(term_p_x, x, i=0, j=0) + \
                  dde.grad.jacobian(term_p_y, x, i=0, j=1)

    eq3_hole = div_Jp_term / S_cont + RG_term

    return [eq1_poisson, eq2_electron, eq3_hole]


# ======================================================================
# BOUNDARY CONDITIONS (QFP Formulation)
# ======================================================================
cathode_x_start = 40.0 * 1e-4
cathode_x_end = 60.0 * 1e-4

V_bi_cathode = V_t * np.log(N_D_nplus / n_i)
V_bi_anode = -V_t * np.log(N_A_pplus / n_i)

print(f"Built-in voltage (QFP): {V_bi_cathode - V_bi_anode:.4f} V")


def on_cathode(x, on_boundary):
    is_on_y = np.isclose(x[1], 0, atol=1e-8)
    is_on_x = (x[0] >= cathode_x_start) & (x[0] <= cathode_x_end)
    return on_boundary and is_on_y and is_on_x


def on_anode(x, on_boundary):
    return on_boundary and np.isclose(x[1], y_total_cm, atol=1e-8)


def get_bcs_robust(V_bias=0.0):
    """
    Returns boundary conditions for the QFP formulation [psi, phi_n, phi_p]
    """

    # --- Cathode (V_bias) ---
    psi_cathode_val = V_bi_cathode + V_bias
    phin_cathode_val = V_bias
    phip_cathode_val = V_bias

    bc_psi_cathode = dde.icbc.DirichletBC(geom, lambda x: psi_cathode_val, on_cathode, component=0)
    bc_phin_cathode = dde.icbc.DirichletBC(geom, lambda x: phin_cathode_val, on_cathode, component=1)
    bc_phip_cathode = dde.icbc.DirichletBC(geom, lambda x: phip_cathode_val, on_cathode, component=2)

    # --- Anode (GND = 0V) ---
    psi_anode_val = V_bi_anode  # V_bias = 0
    phin_anode_val = 0.0
    phip_anode_val = 0.0

    bc_psi_anode = dde.icbc.DirichletBC(geom, lambda x: psi_anode_val, on_anode, component=0)
    bc_phin_anode = dde.icbc.DirichletBC(geom, lambda x: phin_anode_val, on_anode, component=1)
    bc_phip_anode = dde.icbc.DirichletBC(geom, lambda x: phin_anode_val, on_anode, component=2)

    return [bc_psi_cathode, bc_phin_cathode, bc_phip_cathode,
            bc_psi_anode, bc_phin_anode, bc_phip_anode]


# ======================================================================
# CURRENT CALCULATION (QFP Formulation)
# ======================================================================

def J_y_total_robust(x, u):
    """Calculate total vertical current density from QFP variables"""
    psi = u[:, 0:1]
    phi_n = u[:, 1:2]
    phi_p = u[:, 2:3]

    exp_arg_n = torch.clamp((psi - phi_n) / V_t, -80, 80)
    exp_arg_p = torch.clamp((phi_p - psi) / V_t, -80, 80)

    n_raw = n_i * torch.exp(exp_arg_n)
    p_raw = n_i * torch.exp(exp_arg_p)

    n = torch.clamp(n_raw, max=N_max * 1.5)
    p = torch.clamp(p_raw, max=N_max * 1.5)

    # Add floor for stability
    n_floor = n + NUMERICAL_FLOOR
    p_floor = p + NUMERICAL_FLOOR

    dphin_y = dde.grad.jacobian(phi_n, x, i=0, j=1)
    dphip_y = dde.grad.jacobian(phi_p, x, i=0, j=1)

    Jn_y = -q * mu_n * n_floor * dphin_y
    Jp_y = -q * mu_p * p_floor * dphip_y

    return Jn_y + Jp_y


def calculate_current(model, n_points_x=50):
    """
    Calculate terminal current by integrating J_y at the.
    """
    x_contact = np.linspace(0, width_cm, n_points_x)
    y_contact = np.full_like(x_contact, y_total_cm)
    contact_points = np.vstack((x_contact, y_contact)).T

    J_y_values = model.predict(contact_points, operator=J_y_total_robust)
    J_y_values = J_y_values.flatten()

    current_per_cm = -spi.trapezoid(J_y_values, x_contact)

    # If current is NaN (from a failed sim), return 0 to avoid crashing
    if np.isnan(current_per_cm):
        print("  WARNING: Current calculation resulted in NaN. Returning 0.")
        return 0.0

    return current_per_cm


# ======================================================================
# SOLVER (ADAM Only, Multi-Stage LR, No L-BFGS)
# ======================================================================

# Network outputs [psi, phi_n, phi_p]
layer_sizes = [2] + [80] * 4 + [3]
net = dde.nn.FNN(layer_sizes, "tanh", "Glorot normal")

# Store the V=0 dark model globally



def solve_photodiode_robust(V_bias, iters_stage1, iters_stage2, iters_stage3, initial_weights=None):
    """
    Solve for specific bias point using robust QFP model
    and manual loss weighting.
    """
    global net  # Use the global network

    gen_status = "DARK" if optical_gen.is_dark else f"LIGHT (G₀={optical_gen.G0:.2e})"
    print("-" * 60)
    print(f"Solving for V_bias = {V_bias:.3f} V, {gen_status}")
    print(f"  Continuity Scaler: {optical_gen.get_continuity_scaler():.2e}")

    bcs = get_bcs_robust(V_bias)
    data = dde.data.PDE(
        geom,
        pde_system_robust,
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

        # Re-initialize network if not using transfer learning
        def init_weights(m):
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_normal_(m.weight)
                m.bias.data.fill_(0.0)

        net.apply(init_weights)

    # --- MANUAL LOSS WEIGHTING ---
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
# MAIN EXECUTION (Curriculum Learning)
# ======================================================================

if __name__ == "__main__":

    # **FIX**: Declare global *before* any assignment
    global weights_dark_0V

    # Reduced voltage sweep for faster demonstration
    V_sweep = np.array([-0.5, 0.0, 0.5])

    currents_dark = []
    currents_light = []

    # --- Set iterations ---
    # More iterations for the first V=0 solve
    iters_s1_first = 20000
    iters_s2_first = 10000
    iters_s3_first = 5000

    # Fewer iterations for transfer learning sweeps
    iters_s1_transfer = 8000
    iters_s2_transfer = 4000
    iters_s3_transfer = 2000

    # --- CURRICULUM TRICK: Solve for V=0.0 DARK first ---
    print("\n" + "=" * 70)
    print("PRE-TRAINING: Solving V=0.0 DARK for stable initial guess")
    print("=" * 70)
    optical_gen.set_dark()

    model_dark_0V, weights_dark_0V = solve_photodiode_robust(
        0.0, iters_s1_first, iters_s2_first, iters_s3_first, initial_weights=None
    )
    I_dark_0V_calc = calculate_current(model_dark_0V)
    print(f"  >>> I_dark(0.000V) = {I_dark_0V_calc:.3e} A/cm")

    # Store the V=0 results
    V_sweep_dark = [0.0]
    currents_dark_vals = [I_dark_0V_calc]

    # --- DARK I-V SWEEP (using V=0 model) ---
    print("\n" + "=" * 70)
    print("ROBUST DARK I-V SWEEP (QFP Formulation)")
    print("=" * 70)

    for V in V_sweep:
        if V == 0.0:  # Already solved
            continue

        V_sweep_dark.append(V)
        # **FIX**: Always use the stable weights_dark_0V as the guess
        model, _ = solve_photodiode_robust(
            V, iters_s1_transfer, iters_s2_transfer, iters_s3_transfer,
            initial_weights=weights_dark_0V
        )
        current = calculate_current(model)
        print(f"  >>> I_dark({V:.3f}V) = {current:.3e} A/cm")
        currents_dark_vals.append(current)

    # Sort results for plotting
    dark_results = sorted(zip(V_sweep_dark, currents_dark_vals))
    V_sweep_dark_sorted = [v for v, i in dark_results]
    currents_dark_sorted = [i for v, i in dark_results]

    # --- LIGHT I-V SWEEP (using V=0 model) ---
    print("\n" + "=" * 70)
    print("ROBUST LIGHT I-V SWEEP (850nm, 1 mW/cm²)")
    print("=" * 70)

    optical_gen.set_light(850, P_opt_W_cm2=0.001)

    V_sweep_light = []
    currents_light_vals = []
    for V in V_sweep:
        V_sweep_light.append(V)
        # **FIX**: Always use the stable weights_dark_0V as the guess
        model, _ = solve_photodiode_robust(
            V, iters_s1_transfer, iters_s2_transfer, iters_s3_transfer,
            initial_weights=weights_dark_0V
        )
        current = calculate_current(model)
        print(f"  >>> I_light({V:.3f}V) = {current:.3e} A/cm")
        currents_light_vals.append(current)

    # Sort results for plotting
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
    plt.title('Photodiode I-V Characteristic (Robust QFP v6)', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig("photodiode_IV_robust.png", dpi=150)
    print("✓ Saved: photodiode_IV_robust.png")

    # --- QE SWEEP (Shortened for demo) ---
    print("\n" + "=" * 70)
    print("ROBUST QUANTUM EFFICIENCY SWEEP (V=0)")
    print("=" * 70)

    # We already have I_dark_0V and the V=0 dark model
    print(f"Using pre-calculated I_dark(0V) = {I_dark_0V_calc:.3e} A/cm")

    wavelengths_nm = np.array([400, 700, 950])
    photocurrents = []
    EQE_values = []
    P_opt = 0.001  # 1 mW/cm²

    for lam_nm in wavelengths_nm:
        print("-" * 60)
        optical_gen.set_light(lam_nm, P_opt_W_cm2=P_opt)

        # Use transfer learning from the dark V=0 model
        model_light, _ = solve_photodiode_robust(
            0.0, iters_s1_transfer, iters_s2_transfer, iters_s3_transfer,
            initial_weights=weights_dark_0V
        )

        I_light = calculate_current(model_light)
        I_photo = I_light - I_dark_0V_calc
        photocurrents.append(I_photo)

        # Calculate EQE
        h_J = 6.626e-34
        c_m_s = 2.998e8
        lambda_m = lam_nm * 1e-9
        E_photon_J = h_J * c_m_s / lambda_m
        Phi_0 = P_opt / E_photon_J

        photons_in_per_cm = Phi_0 * width_cm
        electrons_out_per_cm = abs(I_photo) / q

        EQE = (electrons_out_per_cm / photons_in_per_cm) if photons_in_per_cm > 0 else 0
        EQE_values.append(EQE * 100)

        print(f"  >>> λ={lam_nm}nm: I_photo={I_photo:.3e} A/cm, EQE={EQE * 100:.1f}%")

    # --- PLOT QE ---
    plt.figure(figsize=(10, 7))
    plt.plot(wavelengths_nm, EQE_values, 'go-', linewidth=2, markersize=8)
    plt.xlabel('Wavelength (nm)', fontsize=14)
    plt.ylabel('EQE (%)', fontsize=14)
    plt.title('Photodiode Spectral Response (Robust QFP v6)', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 100)  # Clamp plot to physical limits
    plt.tight_layout()
    plt.savefig("photodiode_QE_robust.png", dpi=150)
    print("✓ Saved: photodiode_QE_robust.png")

    print("\n✅ Robust simulation run complete!")