import os
import logging
import time

# --- 1. CRITICAL: FORCE CPU BEFORE IMPORTING DEEPXDE ---
import torch

# HACK: Overwrite the MPS availability check to return False.
try:
    torch.backends.mps.is_available = lambda: False
    print("✓ Patched torch.backends.mps.is_available to False (Forcing CPU)")
except Exception as e:
    print(f"Warning: Could not patch MPS availability: {e}")

import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt
import copy
import scipy.integrate as spi
import gc

# --- 2. CONFIGURATION ---
dde.config.set_default_float("float32")
dde.config.set_random_seed(42)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

device = torch.device("cpu")
logger.info(f"Using device: {device} (Forced via patch)")

def setup_directories():
    os.makedirs("results", exist_ok=True)
    os.makedirs("results/plots", exist_ok=True)

def save_plot(fig, filename):
    filepath = os.path.join("results/plots", filename)
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"✓ Saved plot to {filepath}")

# --- 3. PHYSICAL CONSTANTS ---
class PhysicsConstants:
    q = 1.602e-19
    k_B = 1.381e-23
    eps_0 = 8.854e-14
    T = 300.0
    V_t = k_B * T / q
    
    eps_r = 11.7
    epsilon = eps_r * eps_0
    n_i = 1.0e10
    
    mu_n = 1400.0
    mu_p = 450.0
    
    tau_n = 1.0e-6
    tau_p = 1.0e-6
    
    R_reflectance = 0.3

class Scaling:
    L_scale = 1e-4  # 1 micron = 1e-4 cm
    N_ref = 1e18
    
    scale_poisson = (PhysicsConstants.q * N_ref) / PhysicsConstants.epsilon
    
    # Dynamic Transport Scale (~1e29)
    _transport_mag = PhysicsConstants.mu_n * PhysicsConstants.V_t * N_ref / (L_scale**2)
    scale_continuity = _transport_mag 
    
    @staticmethod
    def log_scales():
        logger.info(f"Scale Poisson: {Scaling.scale_poisson:.2e}")
        logger.info(f"Scale Continuity: {Scaling.scale_continuity:.2e}")

# --- 4. GEOMETRY & DOPING ---
class DeviceGeometry:
    width_um = 100.0
    t_n_plus = 1.0
    t_p = 30.0
    t_p_plus = 5.0
    y_total_um = t_n_plus + t_p + t_p_plus
    y_j1 = t_n_plus
    y_j2 = t_n_plus + t_p
    cathode_x_range = (40.0, 60.0)
    
    N_D_cathode = 1e18
    N_A_bulk = 1e15
    N_A_anode = 1e18

    @staticmethod
    def doping_profile_tensor(x_um_tensor):
        y = x_um_tensor[:, 1:2]
        k = 20.0
        sig_j1 = 0.5 * (1 + torch.tanh(k * (y - DeviceGeometry.y_j1)))
        sig_j2 = 0.5 * (1 + torch.tanh(k * (y - DeviceGeometry.y_j2)))
        n_region = 1.0 - sig_j1
        p_region = sig_j1 * (1.0 - sig_j2)
        pplus_region = sig_j2
        return (DeviceGeometry.N_D_cathode * n_region) - \
               (DeviceGeometry.N_A_bulk * p_region) - \
               (DeviceGeometry.N_A_anode * pplus_region)

# --- 5. PHYSICS ENGINE ---
class PhotodiodePhysics:
    def __init__(self, voltage_bias, optical_params=None, equilibrium_mode=False):
        self.V_bias = voltage_bias
        self.optical = optical_params 
        self.equilibrium_mode = equilibrium_mode
        
        # Built-in potentials
        self.psi_n_eq = PhysicsConstants.V_t * np.log(DeviceGeometry.N_D_cathode / PhysicsConstants.n_i)
        self.psi_p_bulk_eq = -PhysicsConstants.V_t * np.log(DeviceGeometry.N_A_bulk / PhysicsConstants.n_i)
        self.psi_p_anode_eq = -PhysicsConstants.V_t * np.log(DeviceGeometry.N_A_anode / PhysicsConstants.n_i)

    def pde(self, x, u):
        psi = u[:, 0:1]
        phi_n = u[:, 1:2]
        phi_p = u[:, 2:3]
        
        limit = 20.0
        
        # FIX: Always use network outputs for Quasi-Fermi levels
        phi_n_phys = phi_n
        phi_p_phys = phi_p

        n = PhysicsConstants.n_i * torch.exp(torch.clamp((psi - phi_n_phys) / PhysicsConstants.V_t, -limit, limit))
        p = PhysicsConstants.n_i * torch.exp(torch.clamp((phi_p_phys - psi) / PhysicsConstants.V_t, -limit, limit))
        
        # Poisson Equation
        N_net = DeviceGeometry.doping_profile_tensor(x)
        rho = PhysicsConstants.q * (p - n + N_net)
        
        grad_psi_x = dde.grad.jacobian(psi, x, i=0, j=0)
        grad_psi_y = dde.grad.jacobian(psi, x, i=0, j=1)
        lap_psi = (dde.grad.jacobian(grad_psi_x, x, i=0, j=0) + dde.grad.jacobian(grad_psi_y, x, i=0, j=1)) * (1.0 / Scaling.L_scale)**2
        
        res_poisson = (PhysicsConstants.epsilon * lap_psi + rho) / Scaling.scale_poisson

        # Symmetry Regularization (Soft guide for 1D bulk behavior)
        res_symmetry = grad_psi_x * 0.1 

        # --- Equilibrium Mode: Gradient Penalty Strategy ---
        if self.equilibrium_mode:
            # Force flat Quasi-Fermi levels (Zero Current condition)
            grad_phin_x = dde.grad.jacobian(phi_n, x, i=0, j=0)
            grad_phin_y = dde.grad.jacobian(phi_n, x, i=0, j=1)
            grad_phip_x = dde.grad.jacobian(phi_p, x, i=0, j=0)
            grad_phip_y = dde.grad.jacobian(phi_p, x, i=0, j=1)
            
            # Strong penalty for any gradient
            res_electron = (grad_phin_x**2 + grad_phin_y**2) * 1e3
            res_hole = (grad_phip_x**2 + grad_phip_y**2) * 1e3
            
            return [res_poisson, res_electron, res_hole, res_symmetry]
        
        # --- Non-Equilibrium Physics ---
        n_i = PhysicsConstants.n_i
        U_srh = (n * p - n_i**2) / (PhysicsConstants.tau_p * (n + n_i) + PhysicsConstants.tau_n * (p + n_i))
        
        if self.optical and self.optical['G0'] > 0:
            y_cm = x[:, 1:2] * Scaling.L_scale
            G_opt = self.optical['G0'] * torch.exp(-self.optical['alpha'] * y_cm)
        else:
            G_opt = 0.0
            
        R_net = U_srh - G_opt
        inv_L = 1.0 / Scaling.L_scale
        
        grad_phin_x = dde.grad.jacobian(phi_n, x, i=0, j=0) * inv_L
        grad_phin_y = dde.grad.jacobian(phi_n, x, i=0, j=1) * inv_L
        Fn_x = -PhysicsConstants.mu_n * n * grad_phin_x
        Fn_y = -PhysicsConstants.mu_n * n * grad_phin_y
        div_Fn = (dde.grad.jacobian(Fn_x, x, i=0, j=0) + dde.grad.jacobian(Fn_y, x, i=0, j=1)) * inv_L
        res_electron = (div_Fn - R_net) / Scaling.scale_continuity

        grad_phip_x = dde.grad.jacobian(phi_p, x, i=0, j=0) * inv_L
        grad_phip_y = dde.grad.jacobian(phi_p, x, i=0, j=1) * inv_L
        Fp_x = -PhysicsConstants.mu_p * p * grad_phip_x
        Fp_y = -PhysicsConstants.mu_p * p * grad_phip_y
        div_Fp = (dde.grad.jacobian(Fp_x, x, i=0, j=0) + dde.grad.jacobian(Fp_y, x, i=0, j=1)) * inv_L
        res_hole = (div_Fp + R_net) / Scaling.scale_continuity
        
        return [res_poisson, res_electron, res_hole, res_symmetry]

    def get_boundary_conditions(self):
        geom = dde.geometry.Rectangle([0, 0], [DeviceGeometry.width_um, DeviceGeometry.y_total_um])
        
        def on_cathode(x, on_boundary):
            return on_boundary and np.isclose(x[1], 0) and \
                   (x[0] >= DeviceGeometry.cathode_x_range[0]) and \
                   (x[0] <= DeviceGeometry.cathode_x_range[1])

        # BCs must match physical conditions
        val_psi_cat = self.psi_n_eq + self.V_bias
        val_phi_cat = self.V_bias
        
        bc_psi_c = dde.icbc.DirichletBC(geom, lambda x: val_psi_cat, on_cathode, component=0)
        bc_phin_c = dde.icbc.DirichletBC(geom, lambda x: val_phi_cat, on_cathode, component=1)
        bc_phip_c = dde.icbc.DirichletBC(geom, lambda x: val_phi_cat, on_cathode, component=2)
        
        def on_anode(x, on_boundary):
            return on_boundary and np.isclose(x[1], DeviceGeometry.y_total_um)
            
        val_psi_anode = self.psi_p_anode_eq
        val_phi_anode = 0.0
        
        bc_psi_a = dde.icbc.DirichletBC(geom, lambda x: val_psi_anode, on_anode, component=0)
        bc_phin_a = dde.icbc.DirichletBC(geom, lambda x: val_phi_anode, on_anode, component=1)
        bc_phip_a = dde.icbc.DirichletBC(geom, lambda x: val_phi_anode, on_anode, component=2)
        
        # Insulation (Neumann BCs)
        def on_insulator(x, on_boundary):
            is_left = np.isclose(x[0], 0)
            is_right = np.isclose(x[0], DeviceGeometry.width_um)
            is_top_passivation = np.isclose(x[1], 0) and not (
                x[0] >= DeviceGeometry.cathode_x_range[0] and x[0] <= DeviceGeometry.cathode_x_range[1]
            )
            return on_boundary and (is_left or is_right or is_top_passivation)

        bc_neumann_psi = dde.icbc.NeumannBC(geom, lambda x: 0, on_insulator, component=0)
        bc_neumann_phin = dde.icbc.NeumannBC(geom, lambda x: 0, on_insulator, component=1)
        bc_neumann_phip = dde.icbc.NeumannBC(geom, lambda x: 0, on_insulator, component=2)
        
        return [bc_psi_c, bc_phin_c, bc_phip_c, bc_psi_a, bc_phin_a, bc_phip_a,
                bc_neumann_psi, bc_neumann_phin, bc_neumann_phip]

# --- 6. IMPROVED ANALYTICAL GUESS ---
def analytical_guess(x_numpy, V_bias=0.0):
    y = x_numpy[:, 1:2]
    
    psi_n = PhysicsConstants.V_t * np.log(DeviceGeometry.N_D_cathode / PhysicsConstants.n_i)
    psi_p = -PhysicsConstants.V_t * np.log(DeviceGeometry.N_A_bulk / PhysicsConstants.n_i)
    psi_pp = -PhysicsConstants.V_t * np.log(DeviceGeometry.N_A_anode / PhysicsConstants.n_i)
    
    k = 20.0
    sig_1 = 0.5 * (1 + np.tanh(k * (y - DeviceGeometry.y_j1)))
    sig_2 = 0.5 * (1 + np.tanh(k * (y - DeviceGeometry.y_j2)))
    
    # Potential guess includes bias
    psi_guess = (psi_n + V_bias) * (1 - sig_1) + psi_p * (sig_1 * (1 - sig_2)) + psi_pp * sig_2
    
    # Quasi-Fermi guess: Smooth transition from V_bias (cathode) to 0 (anode)
    # If V_bias=0, this naturally gives 0 everywhere.
    phi_n_guess = V_bias * (1 - sig_1 * 0.5) 
    phi_p_guess = V_bias * (sig_2 * 0.5)
    
    return np.hstack((psi_guess, phi_n_guess, phi_p_guess))

# --- 7. MODEL FACTORY ---
def create_model(V_bias, optical_params=None, initial_weights=None):
    is_dark = (optical_params is None)
    eq_mode = (V_bias == 0.0 and is_dark)
    
    physics = PhotodiodePhysics(V_bias, optical_params, equilibrium_mode=eq_mode)
    bcs = physics.get_boundary_conditions()
    geom = dde.geometry.Rectangle([0, 0], [DeviceGeometry.width_um, DeviceGeometry.y_total_um])
    
    def junction_sampler(n_points):
        x = np.random.uniform(0, DeviceGeometry.width_um, n_points)
        n_uni = int(0.4 * n_points)
        n_j1 = int(0.3 * n_points)
        n_j2 = n_points - n_uni - n_j1
        y_uni = np.random.uniform(0, DeviceGeometry.y_total_um, n_uni)
        y_j1_pts = np.random.normal(DeviceGeometry.y_j1, 0.5, n_j1)
        y_j2_pts = np.random.normal(DeviceGeometry.y_j2, 0.5, n_j2)
        y = np.concatenate([y_uni, y_j1_pts, y_j2_pts])
        y = np.clip(y, 0, DeviceGeometry.y_total_um)
        return np.column_stack((x, y)).astype(np.float32)

    data = dde.data.PDE(
        geom, physics.pde, bcs, num_domain=0, num_boundary=400, anchors=junction_sampler(3000)
    )
    
    def feature_transform(x):
        x_norm = torch.zeros_like(x)
        x_norm[:, 0] = (x[:, 0] - DeviceGeometry.width_um/2) / (DeviceGeometry.width_um/2)
        x_norm[:, 1] = (x[:, 1] - DeviceGeometry.y_total_um/2) / (DeviceGeometry.y_total_um/2)
        return x_norm

    net = dde.nn.FNN([2] + [64] * 6 + [3], "tanh", "Glorot normal")
    net.apply_feature_transform(feature_transform)
    net.to(device)
    
    model = dde.Model(data, net)
    if initial_weights:
        model.net.load_state_dict(initial_weights)
        logger.info(f"Loaded weights (Equilibrium Mode: {eq_mode})")
    return model

# --- 8. TRAINING ---
def pretrain_analytical(model, V_bias, iters=5000):
    logger.info(f">>> Pre-training on Analytical Guess (V={V_bias}V)...")
    geom = model.data.geom
    X_train = geom.random_points(2000)
    Y_train = analytical_guess(X_train, V_bias)
    X_tensor = torch.from_numpy(X_train).float().to(device)
    Y_tensor = torch.from_numpy(Y_train).float().to(device)
    model.net.to(device)
    optimizer = torch.optim.Adam(model.net.parameters(), lr=1e-3)
    model.net.train()
    for i in range(iters):
        optimizer.zero_grad()
        y_pred = model.net(X_tensor)
        loss = torch.mean(torch.square(y_pred - Y_tensor))
        loss.backward()
        optimizer.step()
    logger.info(">>> Pre-training complete.")

def train_physics(model, adam_iters=10000, bfgs=True):
    loss_weights = [100.0, 1.0, 1.0, 1.0] + [50.0]*9
    model.compile("adam", lr=1e-3, loss_weights=loss_weights)
    model.train(iterations=adam_iters, display_every=1000)
    if bfgs:
        logger.info("   >>> Refining with L-BFGS...")
        model.compile("L-BFGS", loss_weights=loss_weights)
        model.train()

# --- 9. POST-PROCESSING & VERIFICATION ---
def verify_solution_quality(model, V_bias, label):
    logger.info(f"--- Verifying {label} solution at {V_bias}V ---")
    
    # Calculate target potentials
    psi_n_eq = PhysicsConstants.V_t * np.log(DeviceGeometry.N_D_cathode / PhysicsConstants.n_i)
    psi_p_eq = -PhysicsConstants.V_t * np.log(DeviceGeometry.N_A_anode / PhysicsConstants.n_i)
    
    target_psi_cat = psi_n_eq + V_bias
    target_psi_ano = psi_p_eq
    
    # Targets for Quasi-Fermi
    target_phi_cat = V_bias
    target_phi_ano = 0.0
    
    # Sample points
    cat_pt = np.array([[50.0, 0.001]]).astype(np.float32)
    ano_pt = np.array([[50.0, DeviceGeometry.y_total_um - 0.001]]).astype(np.float32)
    
    u_cat = model.predict(cat_pt)
    u_ano = model.predict(ano_pt)
    
    logger.info(f"  Cathode Psi: {u_cat[0,0]:.3f} V (Target: {target_psi_cat:.3f})")
    logger.info(f"  Cathode Phi_n: {u_cat[0,1]:.3f} V (Target: {target_phi_cat:.3f})")
    logger.info(f"  Anode Psi:   {u_ano[0,0]:.3f} V (Target: {target_psi_ano:.3f})")
    logger.info(f"  Anode Phi_p: {u_ano[0,2]:.3f} V (Target: {target_phi_ano:.3f})")

def calculate_current(model):
    n_pts = 500
    x_vals_um = np.linspace(0, DeviceGeometry.width_um, n_pts)
    eps = 0.001 * DeviceGeometry.y_total_um
    y_vals_um = np.full_like(x_vals_um, DeviceGeometry.y_total_um - eps)
    coords = np.column_stack((x_vals_um, y_vals_um))
    
    try:
        x_tensor = torch.tensor(coords, dtype=torch.float32, device=device)
        x_tensor.requires_grad = True
        model.net.to(device)
        u = model.net(x_tensor)
        psi = u[:, 0:1]; phi_n = u[:, 1:2]; phi_p = u[:, 2:3]
        limit = 20.0
        n = PhysicsConstants.n_i * torch.exp(torch.clamp((psi - phi_n) / PhysicsConstants.V_t, -limit, limit))
        p = PhysicsConstants.n_i * torch.exp(torch.clamp((phi_p - psi) / PhysicsConstants.V_t, -limit, limit))
        
        grad_phin = torch.autograd.grad(phi_n, x_tensor, torch.ones_like(phi_n), create_graph=False, retain_graph=True)[0]
        grad_phip = torch.autograd.grad(phi_p, x_tensor, torch.ones_like(phi_p), create_graph=False)[0]
        
        dphin_dy_cm = grad_phin[:, 1:2] * (1.0 / Scaling.L_scale)
        dphip_dy_cm = grad_phip[:, 1:2] * (1.0 / Scaling.L_scale)
        Jn_y = -PhysicsConstants.q * PhysicsConstants.mu_n * n * dphin_dy_cm
        Jp_y = -PhysicsConstants.q * PhysicsConstants.mu_p * p * dphip_dy_cm
        J_total = Jn_y + Jp_y
        J_vals = J_total.cpu().detach().numpy().flatten()
        del x_tensor, u, n, p, grad_phin, grad_phip
        gc.collect()
        x_vals_cm = x_vals_um * Scaling.L_scale
        current_A_per_cm_depth = spi.trapezoid(J_vals, x_vals_cm)
        return current_A_per_cm_depth
    except Exception as e:
        logger.error(f"Error in current calculation: {e}")
        return float('nan')

def plot_1d_profiles(model, bias, label):
    n_points = 500
    y = np.linspace(0, DeviceGeometry.y_total_um, n_points)
    x = np.full_like(y, DeviceGeometry.width_um / 2)
    pts = np.column_stack((x, y))
    
    model.net.to(device)
    u = model.predict(pts)
    psi = u[:, 0]; phi_n = u[:, 1]; phi_p = u[:, 2]
    limit = 20.0
    n = PhysicsConstants.n_i * np.exp(np.clip((psi - phi_n)/PhysicsConstants.V_t, -limit, limit))
    p = PhysicsConstants.n_i * np.exp(np.clip((phi_p - psi)/PhysicsConstants.V_t, -limit, limit))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    ax1.plot(y, psi, label='Psi')
    ax1.plot(y, phi_n, '--', label='Phi_n')
    ax1.plot(y, phi_p, '--', label='Phi_p')
    ax1.set_xlabel("Depth (um)"); ax1.set_ylabel("Potential (V)")
    ax1.legend(); ax1.grid(True)
    ax2.semilogy(y, n, label='n')
    ax2.semilogy(y, p, label='p')
    ax2.axhline(PhysicsConstants.n_i, color='k', linestyle=':', label='n_i')
    ax2.set_xlabel("Depth (um)"); ax2.set_ylabel("Carrier Density (cm^-3)")
    ax2.legend(); ax2.grid(True)
    save_plot(fig, f"1D_Profiles_{label}_{bias}V.png")

def plot_2d_contour(model, bias, label, var_name='psi'):
    x = np.linspace(0, DeviceGeometry.width_um, 100)
    y = np.linspace(0, DeviceGeometry.y_total_um, 200)
    X, Y = np.meshgrid(x, y)
    pts = np.vstack((X.ravel(), Y.ravel())).T
    
    model.net.to(device)
    u = model.predict(pts)
    
    if var_name == 'psi':
        Z = u[:, 0].reshape(X.shape)
        title = "Potential (V)"
    else:
        psi = u[:, 0]; phi_n = u[:, 1]; phi_p = u[:, 2]
        limit = 20.0
        if var_name == 'n':
            val = PhysicsConstants.n_i * np.exp(np.clip((psi - phi_n)/PhysicsConstants.V_t, -limit, limit))
            title = "Electron Conc (cm^-3)"
        else:
            val = PhysicsConstants.n_i * np.exp(np.clip((phi_p - psi)/PhysicsConstants.V_t, -limit, limit))
            title = "Hole Conc (cm^-3)"
        Z = np.log10(val + 1e-10).reshape(X.shape)
        title = "Log10 " + title

    fig, ax = plt.subplots(figsize=(6, 5))
    c = ax.contourf(X, Y, Z, levels=50, cmap='viridis')
    plt.colorbar(c, label=title)
    ax.set_title(f'{var_name} Profile ({label}, {bias}V)')
    ax.set_xlabel('Width (um)'); ax.set_ylabel('Depth (um)')
    ax.invert_yaxis()
    save_plot(fig, f"2D_{var_name}_{label}_{bias}V.png")

def plot_iv_curve(voltages, currents_dark, currents_light):
    plt.figure(figsize=(8, 6))
    plt.plot(voltages, np.abs(currents_dark), 'o-', label='Dark')
    if currents_light:
        plt.plot(voltages, np.abs(currents_light), 'o-', label='Light')
    plt.xlabel("Bias Voltage (V)")
    plt.ylabel("Current Magnitude |I| (A/cm)")
    plt.yscale('log')
    plt.title("I-V Characteristic")
    plt.legend(); plt.grid(True)
    save_plot(plt.gcf(), "IV_Curve.png")

def plot_spectral_response(wavelengths, eqes):
    plt.figure(figsize=(8, 6))
    plt.plot(wavelengths, eqes, 'o-')
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("EQE (%)")
    plt.title("Spectral Response (0V)")
    plt.grid(True)
    save_plot(plt.gcf(), "Spectral_Response.png")

# --- 10. MAIN SIMULATION ---
def run_simulation():
    setup_directories()
    Scaling.log_scales()
    
    dark_weights_db = {} 
    voltages = [0.0, -0.5, -1.0]
    
    # === STEP 1: DARK SIMULATION ===
    logger.info("=== STEP 1: DARK CURRENT SIMULATION ===")
    dark_currents = []
    last_weights = None
    
    for i, v in enumerate(voltages):
        logger.info(f"\n--- Solving Dark Condition: V_bias = {v} V ---")
        model = create_model(v, optical_params=None, initial_weights=last_weights)
        
        # Improved Pre-training
        if i == 0: pretrain_analytical(model, v, iters=5000)
        
        train_physics(model, adam_iters=5000 if i==0 else 8000, bfgs=True)
        
        # Verification Step
        verify_solution_quality(model, v, "Dark")
        
        I_dark = calculate_current(model)
        dark_currents.append(I_dark)
        
        plot_1d_profiles(model, v, "Dark")
        plot_2d_contour(model, v, "Dark", 'psi')
        if v == 0.0:
            plot_2d_contour(model, v, "Dark", 'n')
        
        current_weights = copy.deepcopy(model.net.state_dict())
        dark_weights_db[v] = current_weights
        last_weights = current_weights

    # === STEP 2: LIGHT SIMULATION ===
    logger.info("=== STEP 2: ILLUMINATED SIMULATION (850nm) ===")
    wavelength_nm = 850
    E_ph = (6.626e-34 * 3e8) / (wavelength_nm * 1e-9)
    P_opt = 0.1 # Watts/cm^2
    Phi_0 = P_opt / E_ph
    alpha = 600.0
    G0_eff = Phi_0 * (1 - PhysicsConstants.R_reflectance) * alpha
    opt_params_850 = {'G0': G0_eff, 'alpha': alpha}
    
    light_currents = []
    summary_data = []
    
    for i, v in enumerate(voltages):
        logger.info(f"\n--- Solving Light Condition: V_bias = {v} V ---")
        start_weights = dark_weights_db[v]
        model = create_model(v, optical_params=opt_params_850, initial_weights=start_weights)
        
        train_physics(model, adam_iters=5000, bfgs=True)
        
        I_light = calculate_current(model)
        light_currents.append(I_light)
        
        I_photo = abs(I_light - dark_currents[i])
        width_cm = DeviceGeometry.width_um * Scaling.L_scale
        photons_in = Phi_0 * width_cm
        electrons_out = I_photo / PhysicsConstants.q
        eqe = (electrons_out / photons_in) * 100
        summary_data.append((v, I_light, I_photo, eqe))
        
        plot_1d_profiles(model, v, "Light")
        plot_2d_contour(model, v, "Light", 'n')

    print("\n" + "="*50)
    print("FINAL RESULTS SUMMARY (850nm)")
    print("="*50)
    print(f"{'V_bias (V)':<12} | {'I_photo (A/cm)':<18} | {'EQE (%)':<10}")
    print("-" * 50)
    for v, _, i_ph, eqe in summary_data:
        print(f"{v:<12.2f} | {i_ph:<18.4e} | {eqe:<10.2f}")
    print("="*50 + "\n")

    plot_iv_curve(voltages, dark_currents, light_currents)

    # === STEP 3: SPECTRAL RESPONSE ===
    logger.info("=== STEP 3: SPECTRAL RESPONSE (0V) ===")
    wavelengths = [400, 600, 850, 1000]
    eqe_values = []
    base_weights = dark_weights_db[0.0]
    
    for lam in wavelengths:
        logger.info(f"\n--- Solving Wavelength: {lam} nm ---")
        E_ph_lam = (6.626e-34 * 3e8) / (lam * 1e-9)
        Phi_0_lam = P_opt / E_ph_lam
        
        if lam < 400: alpha_lam = 5e5
        elif lam < 600: alpha_lam = 5e3
        elif lam < 900: alpha_lam = 500
        else: alpha_lam = 50
        
        G0_lam = Phi_0_lam * (1 - PhysicsConstants.R_reflectance) * alpha_lam
        opt_params = {'G0': G0_lam, 'alpha': alpha_lam}
        
        model = create_model(0.0, optical_params=opt_params, initial_weights=base_weights)
        
        train_physics(model, adam_iters=3000, bfgs=True)
        
        I_lam = calculate_current(model)
        I_photo_lam = abs(I_lam - dark_currents[0])
        
        width_cm = DeviceGeometry.width_um * Scaling.L_scale
        photons_in = Phi_0_lam * width_cm
        electrons_out = I_photo_lam / PhysicsConstants.q
        eqe = (electrons_out / photons_in) * 100
        eqe_values.append(eqe)
        logger.info(f"Lambda {lam}nm -> EQE: {eqe:.2f}%")

    plot_spectral_response(wavelengths, eqe_values)
    logger.info("All simulations complete.")

if __name__ == "__main__":
    run_simulation()