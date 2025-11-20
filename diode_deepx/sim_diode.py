import deepxde as dde
import torch
import numpy as np
import matplotlib.pyplot as plt
import time
import scipy.integrate as spi
import os

# --- 1. SETUP AND UTILITIES ---

# Set backend and precision
try:
    dde.config.set_default_backend("pytorch")
    print("Using: PyTorch")
except:
    print("Warning: Could not set backend explicitly")

# Device setup
device = torch.device("mps") if torch.backends.mps.is_available() else \
         torch.device("cuda") if torch.cuda.is_available() else \
         torch.device("cpu")
print(f"Using: {device.type.upper()}")
dde.config.set_default_float("float32")

# --- Directory setup for plots ---
def setup_plot_directories(steps):
    for step in steps:
        dir_name = f"plot_step_{step}"
        os.makedirs(dir_name, exist_ok=True)
        print(f"Created/Ensured directory: {dir_name}")

def save_plot(fig, filename, step):
    dir_name = f"plot_step_{step}"
    filepath = os.path.join(dir_name, filename)
    fig.savefig(filepath, dpi=150)
    plt.close(fig)
    print(f"✓ Saved plot to: {filepath}")


# --- 2. PHYSICAL CONSTANTS AND GEOMETRY ---
q = 1.602e-19
k_B = 1.381e-23
eps_0 = 8.854e-14
T = 300.0
V_t = k_B * T / q  # Thermal Voltage

eps_r = 11.7
epsilon = eps_r * eps_0
n_i = 1.0e10
n_i_sq = n_i ** 2

mu_n = 1400.0
mu_p = 450.0

tau_n = 1.0e-6
tau_p = 1.0e-6

# Geometry (n+ / p / p+ structure)
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

# --- Robustness: Characteristic Scales & Floors ---
N_max = max(N_D_nplus, N_A_pplus)
R_thermal = n_i / min(tau_n, tau_p)
NUMERICAL_FLOOR = 1e4 

print(f"Scales: N_max={N_max:.1e}, R_thermal={R_thermal:.1e}")


def doping_profile(x_in):
    """Net doping N_D - N_A (cm⁻³) using Soft Transitions"""
    y = x_in[:, 1:2]
    
    # Sharpness factor for the junction (higher = sharper, but harder to train)
    k = 2000.0 
    
    # Sigmoid transition functions (0 to 1)
    # Transition at y_j1 (n+ to p)
    sig_j1 = 0.5 * (1 + torch.tanh(k * (y - y_j1_cm)))
    # Transition at y_j2 (p to p+)
    sig_j2 = 0.5 * (1 + torch.tanh(k * (y - y_j2_cm)))

    # Construct profile
    # N_D exists below j1
    N_D = N_D_nplus * (1.0 - sig_j1)
    
    # N_A is N_A_p between j1 and j2, and N_A_pplus above j2
    N_A = N_A_p * (sig_j1 - sig_j2) + N_A_pplus * sig_j2 # Approximately

    # Correction for the middle region logic to ensure correct P-region level
    # A safer smooth construction:
    n_region = 1.0 - sig_j1
    p_region = sig_j1 * (1.0 - sig_j2)
    pplus_region = sig_j2
    
    Net_Doping = (N_D_nplus * n_region) - (N_A_p * p_region) - (N_A_pplus * pplus_region)

    return Net_Doping
    
    """Net doping N_D - N_A (cm⁻³)"""
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

# --- 3. OPTICAL GENERATION CLASS (Enhanced) ---

class OpticalGeneration:

    def __init__(self):
        self.G0 = 0.0
        self.alpha = 0.0
        self.is_dark = True
        self.R_thermal = R_thermal

    def set_dark(self):
        self.is_dark = True
        self.G0 = 0.0
        self.alpha = 0.0

    def set_light(self, lambda_nm, P_opt_W_cm2):
        if lambda_nm < 700: alpha_cm = 10000.0
        elif lambda_nm < 900: alpha_cm = 800.0
        elif lambda_nm < 1000: alpha_cm = 100.0
        else: alpha_cm = 10.0

        R_surface = 0.3
        h_J = 6.626e-34
        c_m_s = 2.998e8
        lambda_m = lambda_nm * 1e-9
        E_photon_J = h_J * c_m_s / lambda_m
        Phi_0 = P_opt_W_cm2 / E_photon_J

        self.G0 = Phi_0 * (1.0 - R_surface) * alpha_cm
        self.alpha = alpha_cm
        self.is_dark = False

    def get_generation_term(self, y):
        if self.is_dark:
            return torch.zeros_like(y)
        else:
            return self.G0 * torch.exp(-self.alpha * y)

    def get_continuity_scaler(self):
        return max(self.G0, self.R_thermal, 1e15) # Increased floor for scaling

optical_gen = OpticalGeneration()

   


# Create global instance
optical_gen = OpticalGeneration()

# --- 4. QFP PDE SYSTEM (ROBUST) ---

def pde_system_robust(x, u):
    """
    Robust system in MICRON domain with SOFT CLAMPING and SCALED EQUATIONS.
    """
    psi = u[:, 0:1]
    phi_n = u[:, 1:2]
    phi_p = u[:, 2:3]

    # --- 1. Soft Clamping (Fixes "Dead Zones") ---
    # Instead of hard clamp, we use a smooth approximation if values get too wild.
    # However, for stability, a carefully chosen hard clamp is often safer in early training
    # provided the bounds are wide enough. 
    # Let's stick to wide hard clamp but monitor it.
    # If you want "Soft", use Softplus or LogSumExp, but it complicates the graph.
    # We will widen the clamp to ensure gradients flow.
    
    exp_arg_n = torch.clamp((psi - phi_n) / V_t, -60.0, 60.0)
    exp_arg_p = torch.clamp((phi_p - psi) / V_t, -60.0, 60.0)

    n = n_i * torch.exp(exp_arg_n)
    p = n_i * torch.exp(exp_arg_p)
    
    # --- 2. Doping (Avoid Hard Step Functions!) ---
    # We use the smooth tanh profile defined in the Micron-Scale section.
    N_net = doping_profile(x)
    
    # --- 3. Poisson's Equation (Micron Scaled) ---
    # Convert derivatives from d/dx_um to d/dx_cm
    # Scale Factor = 1e4
    
    dpsi_x = dde.grad.jacobian(psi, x, i=0, j=0)
    dpsi_y = dde.grad.jacobian(psi, x, i=0, j=1)
    d2psi_x2 = dde.grad.jacobian(dpsi_x, x, i=0, j=0)
    d2psi_y2 = dde.grad.jacobian(dpsi_y, x, i=0, j=1)
    
    # Curvature in cm^-2
    laplacian_psi_cm = (d2psi_x2 + d2psi_y2) * (1e8) 
    
    rho = q * (p - n + N_net)

    # SCALING: This fixes the 10^24 mismatch identified in feedback
    scale_poisson = q * N_max 
    eq1_poisson = (epsilon * laplacian_psi_cm + rho) / scale_poisson

    # --- 4. Recombination (Shockley-Read-Hall) ---
    # We ignore Auger for now to ensure stability first.
    U_num = n * p - n_i_sq
    U_den = tau_p * (n + n_i) + tau_n * (p + n_i)
    U = U_num / U_den

    # Generation (Convert um input to cm for physics calculation)
    y_cm = x[:, 1:2] * 1e-4
    G = optical_gen.get_generation_term(y_cm)

    # SCALING: Normalize Continuity
    scale_continuity = N_max / min(tau_n, tau_p)

    # --- 5. Electron Continuity ---
    dphin_x = dde.grad.jacobian(phi_n, x, i=0, j=0) * 1e4
    dphin_y = dde.grad.jacobian(phi_n, x, i=0, j=1) * 1e4

    Fn_x = -mu_n * n * dphin_x
    Fn_y = -mu_n * n * dphin_y
    
    # Divergence in cm^-1
    div_Fn = (dde.grad.jacobian(Fn_x, x, i=0, j=0) + \
              dde.grad.jacobian(Fn_y, x, i=0, j=1)) * 1e4

    eq2_electron = (div_Fn - G + U) / scale_continuity

    # --- 6. Hole Continuity ---
    dphip_x = dde.grad.jacobian(phi_p, x, i=0, j=0) * 1e4
    dphip_y = dde.grad.jacobian(phi_p, x, i=0, j=1) * 1e4

    Fp_x = -mu_p * p * dphip_x
    Fp_y = -mu_p * p * dphip_y
    
    div_Fp = (dde.grad.jacobian(Fp_x, x, i=0, j=0) + \
              dde.grad.jacobian(Fp_y, x, i=0, j=1)) * 1e4

    eq3_hole = (div_Fp - G + U) / scale_continuity

    return [eq1_poisson, eq2_electron, eq3_hole]
    
    """
    Robust system of equations for PINN diode simulation.
    u = [psi, phi_n, phi_p]
    """
    psi = u[:, 0:1]
    phi_n = u[:, 1:2]
    phi_p = u[:, 2:3]

    # --- 1. Carrier Concentrations (With Safety Clamping) ---
    # We clamp the argument of the exponent to prevent float overflow (NaNs)
    # 60.0 corresponds to exp(60) ~ 1e26, which is plenty for semiconductor physics
    exp_arg_n = torch.clamp((psi - phi_n) / V_t, -60.0, 60.0)
    exp_arg_p = torch.clamp((phi_p - psi) / V_t, -60.0, 60.0)

    n = n_i * torch.exp(exp_arg_n)
    p = n_i * torch.exp(exp_arg_p)
    
    # --- 2. Doping and Physics Constants ---
    N_net = doping_profile(x)
    
    # --- 3. Poisson's Equation ---
    # Equation: epsilon * grad^2(psi) + q * (p - n + N_net) = 0
    
    dpsi_x = dde.grad.jacobian(psi, x, i=0, j=0)
    dpsi_y = dde.grad.jacobian(psi, x, i=0, j=1)
    d2psi_x2 = dde.grad.jacobian(dpsi_x, x, i=0, j=0)
    d2psi_y2 = dde.grad.jacobian(dpsi_y, x, i=0, j=1)
    laplacian_psi = d2psi_x2 + d2psi_y2
    
    rho = q * (p - n + N_net)

    # SCALING: Normalize by the maximum expected charge scale (q * N_max).
    # This ensures the residual is Order(1) rather than Order(0.16) or Order(1e18).
    scale_poisson = q * N_max
    eq1_poisson = (epsilon * laplacian_psi + rho) / scale_poisson

    # --- 4. Recombination & Generation ---
    # Shockley-Read-Hall Recombination
    U_num = n * p - n_i_sq
    U_den = tau_p * (n + n_i) + tau_n * (p + n_i)
    U = U_num / U_den

    # Optical Generation
    y = x[:, 1:2]
    G = optical_gen.get_generation_term(y)

    # SCALING: Determine the characteristic rate of carriers (cm^-3 s^-1)
    # Recombination U can reach ~1e24. We must divide by this to prevent gradient explosion.
    scale_continuity = N_max / min(tau_n, tau_p)  # approx 1e18 / 1e-6 = 1e24

    # --- 5. Electron Continuity ---
    # Physics: div(Jn) = q(R - G)  =>  (1/q) div(Jn) - (R - G) = 0
    # Flux Fn = Jn / (-q) = -mu_n * n * grad(phi_n) (Note: Jn has -q, so Fn doesn't have q)
    # Standard form: div(Fn) = G - U  =>  div(Fn) - (G - U) = 0
    
    dphin_x = dde.grad.jacobian(phi_n, x, i=0, j=0)
    dphin_y = dde.grad.jacobian(phi_n, x, i=0, j=1)

    # Electron Flux Vector (Particle Flux, not Current Density)
    # Fn = -mu_n * n * grad(phi_n)
    Fn_x = -mu_n * n * dphin_x
    Fn_y = -mu_n * n * dphin_y
    
    div_Fn = dde.grad.jacobian(Fn_x, x, i=0, j=0) + \
             dde.grad.jacobian(Fn_y, x, i=0, j=1)

    # Equation: div(Flux) - Generation + Recombination = 0
    eq2_electron = (div_Fn - G + U) / scale_continuity

    # --- 6. Hole Continuity ---
    # Physics: div(Jp) = -q(R - G)
    # Flux Fp = Jp / (+q) = -mu_p * p * grad(phi_p)
    # Standard form: div(Fp) = G - U  => div(Fp) - (G - U) = 0
    
    dphip_x = dde.grad.jacobian(phi_p, x, i=0, j=0)
    dphip_y = dde.grad.jacobian(phi_p, x, i=0, j=1)

    # Hole Flux Vector
    Fp_x = -mu_p * p * dphip_x
    Fp_y = -mu_p * p * dphip_y
    
    div_Fp = dde.grad.jacobian(Fp_x, x, i=0, j=0) + \
             dde.grad.jacobian(Fp_y, x, i=0, j=1)

    # Equation: div(Flux) - Generation + Recombination = 0
    eq3_hole = (div_Fp - G + U) / scale_continuity

    return [eq1_poisson, eq2_electron, eq3_hole]

# --- 5. BOUNDARY CONDITIONS (QFP) ---

cathode_x_start = 40.0 * 1e-4
cathode_x_end = 60.0 * 1e-4

# Built-in potentials for Dirichlet BCs
V_bi_cathode = V_t * np.log(N_D_nplus / n_i)
V_bi_anode = -V_t * np.log(N_A_pplus / n_i)


def on_cathode(x, on_boundary):
    is_on_y = np.isclose(x[1], 0, atol=1e-6)
    is_on_x = (x[0] >= cathode_x_start) & (x[0] <= cathode_x_end)
    return on_boundary and is_on_y and is_on_x

def on_anode(x, on_boundary):
    return on_boundary and np.isclose(x[1], y_total_cm, atol=1e-6)

def get_bcs_robust(V_bias=0.0):
    # Cathode (Top, n+)
    psi_cathode_val = V_bi_cathode + V_bias
    phin_cathode_val = V_bias 
    phip_cathode_val = V_bias 

    bc_psi_cathode = dde.icbc.DirichletBC(geom, lambda x: psi_cathode_val, on_cathode, component=0)
    bc_phin_cathode = dde.icbc.DirichletBC(geom, lambda x: phin_cathode_val, on_cathode, component=1)
    bc_phip_cathode = dde.icbc.DirichletBC(geom, lambda x: phip_cathode_val, on_cathode, component=2)

    # Anode (Bottom, p+, GND)
    psi_anode_val = V_bi_anode
    phin_anode_val = 0.0 
    phip_anode_val = 0.0 

    bc_psi_anode = dde.icbc.DirichletBC(geom, lambda x: psi_anode_val, on_anode, component=0)
    bc_phin_anode = dde.icbc.DirichletBC(geom, lambda x: phin_anode_val, on_anode, component=1)
    bc_phip_anode = dde.icbc.DirichletBC(geom, lambda x: phip_anode_val, on_anode, component=2)

    return [bc_psi_cathode, bc_phin_cathode, bc_phip_cathode,
            bc_psi_anode, bc_phin_anode, bc_phip_anode]

# --- 6. CURRENT CALCULATION (ROBUST) ---

def J_y_total_robust(x, u):
    psi = u[:, 0:1]
    phi_n = u[:, 1:2]
    phi_p = u[:, 2:3]

    exp_arg_n = torch.clamp((psi - phi_n) / V_t, -60, 60)
    exp_arg_p = torch.clamp((phi_p - psi) / V_t, -60, 60)

    n = n_i * torch.exp(exp_arg_n)
    p = n_i * torch.exp(exp_arg_p)

    dphin_y = dde.grad.jacobian(phi_n, x, i=0, j=1)
    dphip_y = dde.grad.jacobian(phi_p, x, i=0, j=1)

    # Total Current Density (A/cm^2)
    Jn_y = -q * mu_n * n * dphin_y
    Jp_y = -q * mu_p * p * dphip_y

    return Jn_y + Jp_y

def calculate_current(model, n_points_x=150):
    x_contact = np.linspace(0, width_cm, n_points_x)
    y_contact = np.full_like(x_contact, y_total_cm) # Measure at Anode
    contact_points = np.vstack((x_contact, y_contact)).T

    J_y_values = model.predict(contact_points, operator=J_y_total_robust)
    J_y_values = J_y_values.flatten()

    # Integration: I = Integral(J * dx)
    # Note: J_y is positive UP. Anode current leaves bottom. 
    # Standard convention: Current entering + terminal.
    current_per_cm = -spi.trapezoid(J_y_values, x_contact)
    return current_per_cm


# --- 7. SOLVER FUNCTION (CURRICULUM) ---

def feature_transform(x):
    # x[:,0] is Width (0 to 0.01)
    # x[:,1] is Depth (0 to ~0.0036)
    x_norm = x.clone()
    x_norm[:, 0] = (x[:, 0] - width_cm/2) / (width_cm/2)
    x_norm[:, 1] = (x[:, 1] - y_total_cm/2) / (y_total_cm/2)
    return x_norm

layer_sizes = [2] + [60] * 5 + [3] # Slightly deeper
net = dde.nn.FNN(layer_sizes, "tanh", "Glorot normal")
net.apply_feature_transform(feature_transform)


def solve_photodiode_robust(V_bias, iters_stage1, iters_stage2, iters_stage3, initial_weights=None):
    
    gen_status = "DARK" if optical_gen.is_dark else f"LIGHT"
    print("-" * 60)
    print(f"Solving for V_bias = {V_bias:.3f} V, {gen_status}")

    bcs = get_bcs_robust(V_bias)
    
    # Increased points for better resolution of the smooth junction
    data = dde.data.PDE(
        geom,
        pde_system_robust,
        bcs,
        num_domain=5000,
        num_boundary=1500,
        num_test=1500
    )

    model = dde.Model(data, net)

    if initial_weights:
        model.net.load_state_dict(initial_weights)
    else:
        # Re-initialize network
        def init_weights(m):
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_normal_(m.weight)
                m.bias.data.fill_(0.0)
        net.apply(init_weights)

    # Weights: Heavily penalize BCs to ensure contacts are respected
    loss_weights = [10.0, 1.0, 1.0] + [100.0] * len(bcs)

    # STAGE 1: Fast coarse training
    if iters_stage1 > 0:
        model.compile("adam", lr=1e-3, loss_weights=loss_weights)
        model.train(iterations=iters_stage1, display_every=1000)

    # STAGE 2: Fine tuning
    if iters_stage2 > 0:
        model.compile("adam", lr=2e-4, loss_weights=loss_weights)
        model.train(iterations=iters_stage2, display_every=1000)

    # STAGE 3: L-BFGS (Optional, sometimes unstable with noise, sticking to Adam for robustness here)
    # We use a very low LR Adam instead
    if iters_stage3 > 0:
        model.compile("adam", lr=1e-5, loss_weights=loss_weights)
        model.train(iterations=iters_stage3, display_every=1000)

    import copy
    return model, copy.deepcopy(model.net.state_dict())


# --- 8. VISUALIZATION FUNCTIONS (Step-wise) ---

def plot_1d_profiles(model, V_bias, is_dark, step_name):
    n_points = 500
    y_plot = np.linspace(0, y_total_cm, n_points)
    x_plot = np.full_like(y_plot, width_cm / 2.0)
    plot_domain = np.vstack((x_plot, y_plot)).T

    u_pred = model.predict(plot_domain)
    psi_pred = u_pred[:, 0]
    phi_n_pred = u_pred[:, 1]
    phi_p_pred = u_pred[:, 2]
    
    n_pred = n_i * np.exp((psi_pred - phi_n_pred)/V_t)
    p_pred = n_i * np.exp((phi_p_pred - psi_pred)/V_t)
    
    y_um = y_plot * 1e4

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    title_suffix = "DARK" if is_dark else "ILLUMINATED"
    fig.suptitle(f'1D Center (V={V_bias:.3f}V, {title_suffix})', fontsize=14)

    # Potential
    ax1.plot(y_um, psi_pred, 'k-', lw=2, label='Psi')
    ax1.plot(y_um, phi_n_pred, 'b--', label='Phi_n')
    ax1.plot(y_um, phi_p_pred, 'r--', label='Phi_p')
    ax1.axvline(y_j1_cm * 1e4, color='gray', linestyle=':')
    ax1.axvline(y_j2_cm * 1e4, color='gray', linestyle=':')
    ax1.set_xlabel('Depth (µm)')
    ax1.set_ylabel('Potential (V)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Carriers
    ax2.semilogy(y_um, n_pred, 'b-', lw=2, label='n')
    ax2.semilogy(y_um, p_pred, 'r-', lw=2, label='p')
    ax2.semilogy(y_um, np.full_like(y_um, n_i), 'k:', label='ni')
    ax2.set_xlabel('Depth (µm)')
    ax2.set_ylabel('Concentration (cm⁻³)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    save_plot(fig, f"1D_Profiles_{title_suffix}_V{V_bias:.2f}.png", step_name)

def plot_iv_curve(V_dark, I_dark, V_light, I_light, step_name):
    fig = plt.figure(figsize=(10, 7))
    plt.plot(V_dark, np.array(I_dark) * 1e3, 'bo-', label='Dark')
    if len(V_light) > 0:
        plt.plot(V_light, np.array(I_light) * 1e3, 'ro-', label='Light')
    plt.xlabel('Voltage (V)')
    plt.ylabel('Current (mA/cm)')
    plt.title('I-V Characteristic')
    plt.grid(True)
    plt.legend()
    save_plot(fig, "IV_Characteristic.png", step_name)

def plot_qe_curve(wavelengths, EQE_values, step_name):
    """Plots the External Quantum Efficiency (EQE) spectrum."""
    fig = plt.figure(figsize=(10, 7))
    plt.plot(wavelengths, EQE_values, 'go-', linewidth=2, markersize=8)
    plt.xlabel('Wavelength (nm)', fontsize=14)
    plt.ylabel('EQE (%)', fontsize=14)
    plt.title('Photodiode Spectral Response (Robust QFP)', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 100)
    save_plot(fig, "EQE_Spectral_Response.png", step_name)


def plot_2d_contour(model, V_bias, is_dark, variable_name, component_index, step_name):
    """Generates a 2D contour plot for a single variable (e.g., psi)."""
    grid_size = 50
    x_c = np.linspace(0, width_cm, grid_size)
    y_c = np.linspace(0, y_total_cm, grid_size)
    X, Y = np.meshgrid(x_c, y_c)
    plot_points = np.vstack((X.flatten(), Y.flatten())).T

    u_pred = model.predict(plot_points)
    Z_data = u_pred[:, component_index].reshape(grid_size, grid_size)

    # For carrier concentrations, convert QFP to n or p for visualization
    if variable_name in ['n', 'p']:
        psi_t = torch.tensor(u_pred[:, 0], dtype=torch.float32)
        phin_t = torch.tensor(u_pred[:, 1], dtype=torch.float32)
        phip_t = torch.tensor(u_pred[:, 2], dtype=torch.float32)
        
        n_i_t = torch.tensor(n_i, dtype=torch.float32)
        V_t_t = torch.tensor(V_t, dtype=torch.float32)

        exp_arg_n = torch.clamp((psi_t - phin_t) / V_t_t, -80, 80)
        exp_arg_p = torch.clamp((phip_t - psi_t) / V_t_t, -80, 80)
        
        n_raw = n_i_t * torch.exp(exp_arg_n)
        p_raw = n_i_t * torch.exp(exp_arg_p)

        if variable_name == 'n':
            Z_data = np.log10(np.clip(n_raw.cpu().detach().numpy(), 1e5, N_max * 2)).reshape(grid_size, grid_size)
            label = "log10(n) (cm⁻³)"
        else: # 'p'
            Z_data = np.log10(np.clip(p_raw.cpu().detach().numpy(), 1e5, N_max * 2)).reshape(grid_size, grid_size)
            label = "log10(p) (cm⁻³)"
    else: # psi, phi_n, phi_p
        label = f"{variable_name} (V)"

    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Use tricontourf or pcolormesh
    contour = ax.pcolormesh(X * 1e4, Y * 1e4, Z_data, shading='gouraud', cmap='viridis')
    cbar = fig.colorbar(contour, ax=ax, label=label)
    
    ax.axhline(y_j1_cm * 1e4, color='k', linestyle=':', linewidth=1)
    ax.axhline(y_j2_cm * 1e4, color='k', linestyle=':', linewidth=1)
    
    ax.set_title(f'2D Contour: {variable_name} (V={V_bias:.3f}V, {"DARK" if is_dark else "LIGHT"})')
    ax.set_xlabel('Width (µm)')
    ax.set_ylabel('Depth (µm)')
    ax.set_aspect('equal', adjustable='box')
    
    save_plot(fig, f"2D_Contour_{variable_name}_V_{V_bias:.3f}_{'DARK' if is_dark else 'LIGHT'}.png", step_name)


# --- 9. MAIN EXECUTION: CURRICULUM ---

if __name__ == "__main__":

    # Global variable to store the stable dark V=0 weights
    global weights_dark_0V

    # Define the execution steps for organized plotting
    EXECUTION_STEPS = ["a_pretrain_v0_dark", "b_iv_sweep", "c_qe_sweep"]
    setup_plot_directories(EXECUTION_STEPS)

    # Simulation parameters
    V_sweep_iv = np.array([-0.5,-0.3, 0.0, 0.3, 0.5]) # I-V sweep voltages
    wavelengths_qe = np.array([400, 700, 950, 1050]) # Wavelengths for QE
    P_opt_qe = 0.001 # 1 mW/cm² for light simulations

    # Training iterations (Tuned for demonstration/stability)
    iters_s1_first = 20000
    iters_s2_first = 10000
    iters_s3_first = 5000
    iters_s1_transfer = 8000
    iters_s2_transfer = 4000
    iters_s3_transfer = 2000

    currents_dark_vals = []
    V_sweep_dark = []
    
    # === STEP 1: PRE-TRAINING (V=0.0 DARK) ===
    # This provides the stable initial guess (transfer learning) for all other steps.
    print("\n" + "=" * 70)
    print("STEP 1: PRE-TRAINING (Solving V=0.0 DARK for stable initial guess)")
    print("=" * 70)
    
    optical_gen.set_dark()
    
    # Train the first model from scratch
    model_dark_0V, weights_dark_0V = solve_photodiode_robust(
        0.0, iters_s1_first, iters_s2_first, iters_s3_first, initial_weights=None
    )
    I_dark_0V_calc = calculate_current(model_dark_0V)
    print(f"  >>> I_dark(0.000V) = {I_dark_0V_calc:.3e} A/cm")

    # Store V=0 results
    V_sweep_dark.append(0.0)
    currents_dark_vals.append(I_dark_0V_calc)
    
    # Plot 1D and 2D profiles for the V=0 DARK solution
    plot_1d_profiles(model_dark_0V, 0.0, True, EXECUTION_STEPS[0])
    plot_2d_contour(model_dark_0V, 0.0, True, 'psi', 0, EXECUTION_STEPS[0])
    plot_2d_contour(model_dark_0V, 0.0, True, 'n', 1, EXECUTION_STEPS[0])

    # === STEP 2: DARK & LIGHT I-V SWEEP ===
    currents_light_vals = []
    V_sweep_light = []
    
    # 2.1 Dark I-V Sweep (using V=0 model for transfer)
    print("\n" + "=" * 70)
    print("STEP 2.1: ROBUST DARK I-V SWEEP (Transfer Learning)")
    print("=" * 70)

    for V in V_sweep_iv:
        if V == 0.0: continue
        
        V_sweep_dark.append(V)
        model, _ = solve_photodiode_robust(
            V, iters_s1_transfer, iters_s2_transfer, iters_s3_transfer,
            initial_weights=weights_dark_0V # Transfer Learning!
        )
        current = calculate_current(model)
        print(f"  >>> I_dark({V:.3f}V) = {current:.3e} A/cm")
        currents_dark_vals.append(current)
        
        # Plot 1D profile for the -0.5V dark solution
        if V == -0.5:
            plot_1d_profiles(model, V, True, EXECUTION_STEPS[1])
            plot_2d_contour(model, V, True, 'psi', 0, EXECUTION_STEPS[1])
            
    # Sort results for plotting
    dark_results = sorted(zip(V_sweep_dark, currents_dark_vals))
    V_sweep_dark_sorted = [v for v, i in dark_results]
    currents_dark_sorted = [i for v, i in dark_results]

    # 2.2 Light I-V Sweep (using V=0 model for transfer)
    print("\n" + "=" * 70)
    print("STEP 2.2: ROBUST LIGHT I-V SWEEP (850nm, 1 mW/cm²)")
    print("=" * 70)

    optical_gen.set_light(850, P_opt_W_cm2=P_opt_qe)

    for V in V_sweep_iv:
        V_sweep_light.append(V)
        model, _ = solve_photodiode_robust(
            V, iters_s1_transfer, iters_s2_transfer, iters_s3_transfer,
            initial_weights=weights_dark_0V # Transfer Learning!
        )
        current = calculate_current(model)
        print(f"  >>> I_light({V:.3f}V) = {current:.3e} A/cm")
        currents_light_vals.append(current)

        # Plot 1D profile for the 0V light solution
        if V == 0.0:
            plot_1d_profiles(model, V, False, EXECUTION_STEPS[1])
            plot_2d_contour(model, V, False, 'n', 1, EXECUTION_STEPS[1]) # Carrier profile under light
            
    # Sort results for plotting
    light_results = sorted(zip(V_sweep_light, currents_light_vals))
    V_sweep_light_sorted = [v for v, i in light_results]
    currents_light_sorted = [i for v, i in light_results]
    
    # Plot I-V Curve
    plot_iv_curve(V_sweep_dark_sorted, currents_dark_sorted, 
                  V_sweep_light_sorted, currents_light_sorted, 
                  EXECUTION_STEPS[1])

    # === STEP 3: QUANTUM EFFICIENCY (QE) SWEEP at V=0 ===
    print("\n" + "=" * 70)
    print("STEP 3: ROBUST QUANTUM EFFICIENCY SWEEP (V=0)")
    print("=" * 70)

    photocurrents = []
    EQE_values = []
    h_J = 6.626e-34
    c_m_s = 2.998e8

    for lam_nm in wavelengths_qe:
        optical_gen.set_light(lam_nm, P_opt_W_cm2=P_opt_qe)

        # Use transfer learning from the dark V=0 model
        model_light, _ = solve_photodiode_robust(
            0.0, iters_s1_transfer, iters_s2_transfer, iters_s3_transfer,
            initial_weights=weights_dark_0V
        )

        I_light = calculate_current(model_light)
        I_photo = I_light - I_dark_0V_calc
        photocurrents.append(I_photo)

        # Calculate EQE: EQE = (I_photo/q) / (Phi_inc * W)
        lambda_m = lam_nm * 1e-9
        E_photon_J = h_J * c_m_s / lambda_m
        Phi_0 = P_opt_qe / E_photon_J # Incident flux (photons/cm²/s)

        photons_in_per_cm = Phi_0 * width_cm # Total photons/s per cm of device depth
        electrons_out_per_cm = abs(I_photo) / q

        EQE = (electrons_out_per_cm / photons_in_per_cm) if photons_in_per_cm > 0 else 0
        EQE_values.append(EQE * 100)

        print(f"  >>> λ={lam_nm}nm: I_photo={I_photo:.3e} A/cm, EQE={EQE * 100:.1f}%")

    # Plot QE Curve
    plot_qe_curve(wavelengths_qe, EQE_values, EXECUTION_STEPS[2])

    print("\n✅ Robust, unified simulation run complete!")