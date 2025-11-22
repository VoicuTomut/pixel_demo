import deepxde as dde
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import copy
import scipy.integrate as spi
import logging
import gc

# --- 1. CONFIGURATION & LOGGING ---
dde.config.set_default_float("float32")

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Backend Selection
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

def setup_directories():
    os.makedirs("results", exist_ok=True)
    os.makedirs("results/plots", exist_ok=True)

def save_plot(fig, filename):
    filepath = os.path.join("results/plots", filename)
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved plot to {filepath}")

# --- 2. PHYSICAL CONSTANTS & SCALING ---
class PhysicsConstants:
    q = 1.602e-19       # Elementary charge (C)
    k_B = 1.381e-23     # Boltzmann constant (J/K)
    eps_0 = 8.854e-14   # Vacuum permittivity (F/cm)
    T = 300.0           # Temperature (K)
    V_t = k_B * T / q   # Thermal voltage (~0.0259 V)
    
    # Silicon Parameters
    eps_r = 11.7
    epsilon = eps_r * eps_0
    n_i = 1.0e10        # Intrinsic carrier conc. (cm^-3)
    
    mu_n = 1400.0       # Electron mobility (cm^2/V.s)
    mu_p = 450.0        # Hole mobility
    
    # Recombination (Shockley-Read-Hall)
    tau_n = 1.0e-6      # Electron lifetime (s)
    tau_p = 1.0e-6      # Hole lifetime (s)
    
    # Optical
    R_reflectance = 0.3 # 30% reflection at Si surface

class Scaling:
    """
    Handles conversion between Neural Network inputs (Microns) 
    and Physical equations (cm).
    """
    L_scale = 1e-4  # 1 micron = 1e-4 cm
    
    # Normalization scales for the loss function
    N_ref = 1e18    # Reference doping (cm^-3)
    
    # Scale factors for PDE terms to bring them to Order(1)
    scale_poisson = (PhysicsConstants.q * N_ref) / PhysicsConstants.epsilon
    scale_continuity = N_ref / 1.0e-6

# --- 3. GEOMETRY & DOPING ---
class DeviceGeometry:
    width_um = 100.0
    
    # Layer thicknesses
    t_n_plus = 1.0
    t_p = 30.0
    t_p_plus = 5.0
    
    y_total_um = t_n_plus + t_p + t_p_plus
    
    # Junction depths
    y_j1 = t_n_plus
    y_j2 = t_n_plus + t_p
    
    # Contact Locations (Microns)
    cathode_x_range = (40.0, 60.0) # Top contact (n-type)
    
    # Doping Levels (cm^-3)
    N_D_cathode = 1e18  # n+
    N_A_bulk = 1e15     # p
    N_A_anode = 1e18    # p+

    @staticmethod
    def doping_profile_tensor(x_um_tensor):
        y = x_um_tensor[:, 1:2]
        k = 20.0 # Steepness of junction
        
        sig_j1 = 0.5 * (1 + torch.tanh(k * (y - DeviceGeometry.y_j1)))
        sig_j2 = 0.5 * (1 + torch.tanh(k * (y - DeviceGeometry.y_j2)))
        
        n_region = 1.0 - sig_j1
        p_region = sig_j1 * (1.0 - sig_j2)
        pplus_region = sig_j2
        
        net_doping = (DeviceGeometry.N_D_cathode * n_region) - \
                     (DeviceGeometry.N_A_bulk * p_region) - \
                     (DeviceGeometry.N_A_anode * pplus_region)
        return net_doping

# --- 4. PHYSICS ENGINE ---
class PhotodiodePhysics:
    def __init__(self, voltage_bias, optical_params=None):
        self.V_bias = voltage_bias
        self.optical = optical_params 
        
        # Calculate Built-in Potentials (referenced to intrinsic level psi=0)
        self.psi_n_eq = PhysicsConstants.V_t * np.log(DeviceGeometry.N_D_cathode / PhysicsConstants.n_i)
        self.psi_p_bulk_eq = -PhysicsConstants.V_t * np.log(DeviceGeometry.N_A_bulk / PhysicsConstants.n_i)
        self.psi_p_anode_eq = -PhysicsConstants.V_t * np.log(DeviceGeometry.N_A_anode / PhysicsConstants.n_i)

    def pde(self, x, u):
        psi = u[:, 0:1]
        phi_n = u[:, 1:2]
        phi_p = u[:, 2:3]
        
        # 1. Carrier Statistics (Clamped for Stability)
        limit = 30.0
        n = PhysicsConstants.n_i * torch.exp(torch.clamp((psi - phi_n) / PhysicsConstants.V_t, -limit, limit))
        p = PhysicsConstants.n_i * torch.exp(torch.clamp((phi_p - psi) / PhysicsConstants.V_t, -limit, limit))
        
        # 2. Electrostatics (Poisson)
        N_net = DeviceGeometry.doping_profile_tensor(x)
        rho = PhysicsConstants.q * (p - n + N_net)
        
        grad_psi_x = dde.grad.jacobian(psi, x, i=0, j=0)
        grad_psi_y = dde.grad.jacobian(psi, x, i=0, j=1)
        lap_psi_x = dde.grad.jacobian(grad_psi_x, x, i=0, j=0)
        lap_psi_y = dde.grad.jacobian(grad_psi_y, x, i=0, j=1)
        
        # Laplacian in cm^-2
        lap_psi = (lap_psi_x + lap_psi_y) * (1.0 / Scaling.L_scale)**2
        
        res_poisson = (PhysicsConstants.epsilon * lap_psi + rho) / Scaling.scale_poisson

        # 3. Recombination (SRH)
        n_i = PhysicsConstants.n_i
        U_srh = (n * p - n_i**2) / (
            PhysicsConstants.tau_p * (n + n_i) + 
            PhysicsConstants.tau_n * (p + n_i)
        )
        
        # 4. Optical Generation
        if self.optical and self.optical['G0'] > 0:
            y_cm = x[:, 1:2] * Scaling.L_scale
            G_opt = self.optical['G0'] * torch.exp(-self.optical['alpha'] * y_cm)
        else:
            G_opt = 0.0
            
        R_net = U_srh - G_opt

        # 5. Continuity Equations
        inv_L = 1.0 / Scaling.L_scale
        
        # --- Electron Continuity ---
        # Goal: div(Particle Flux) + R_net = 0
        # Particle Flux Fn = n * v_n = n * (mu_n * grad_phi_n)
        # We define Fn as POSITIVE flux direction to match div(Fn) + R_net = 0 form.
        
        grad_phin_x = dde.grad.jacobian(phi_n, x, i=0, j=0) * inv_L
        grad_phin_y = dde.grad.jacobian(phi_n, x, i=0, j=1) * inv_L
        
        # CORRECTION: Removed negative sign to represent particle flux
        Fn_x = PhysicsConstants.mu_n * n * grad_phin_x
        Fn_y = PhysicsConstants.mu_n * n * grad_phin_y
        
        div_Fn = (dde.grad.jacobian(Fn_x, x, i=0, j=0) + dde.grad.jacobian(Fn_y, x, i=0, j=1)) * inv_L
        
        # CORRECTION: Sign is now (+) R_net.
        res_electron = (div_Fn + R_net) / Scaling.scale_continuity

        # --- Hole Continuity ---
        # Goal: div(Particle Flux) + R_net = 0
        # Particle Flux Fp = p * v_p = p * (-mu_p * grad_phi_p)
        
        grad_phip_x = dde.grad.jacobian(phi_p, x, i=0, j=0) * inv_L
        grad_phip_y = dde.grad.jacobian(phi_p, x, i=0, j=1) * inv_L
        
        # Fp is already negative particle flux definition in standard drift-diffusion?
        # v_p = -mu grad_phi. So Flux = -mu p grad_phi.
        Fp_x = -PhysicsConstants.mu_p * p * grad_phip_x
        Fp_y = -PhysicsConstants.mu_p * p * grad_phip_y
        
        div_Fp = (dde.grad.jacobian(Fp_x, x, i=0, j=0) + dde.grad.jacobian(Fp_y, x, i=0, j=1)) * inv_L
        
        res_hole = (div_Fp + R_net) / Scaling.scale_continuity
        
        return [res_poisson, res_electron, res_hole]

    def get_boundary_conditions(self):
        geom = dde.geometry.Rectangle([0, 0], [DeviceGeometry.width_um, DeviceGeometry.y_total_um])
        
        # Cathode (Top, n+)
        def on_cathode(x, on_boundary):
            return on_boundary and np.isclose(x[1], 0) and \
                   (x[0] >= DeviceGeometry.cathode_x_range[0]) and \
                   (x[0] <= DeviceGeometry.cathode_x_range[1])

        val_psi_cat = self.psi_n_eq + self.V_bias
        val_phi_cat = self.V_bias
        
        bc_psi_c = dde.icbc.DirichletBC(geom, lambda x: val_psi_cat, on_cathode, component=0)
        bc_phin_c = dde.icbc.DirichletBC(geom, lambda x: val_phi_cat, on_cathode, component=1)
        bc_phip_c = dde.icbc.DirichletBC(geom, lambda x: val_phi_cat, on_cathode, component=2)
        
        # Anode (Bottom, p+)
        def on_anode(x, on_boundary):
            return on_boundary and np.isclose(x[1], DeviceGeometry.y_total_um)
            
        val_psi_anode = self.psi_p_anode_eq
        val_phi_anode = 0.0
        
        bc_psi_a = dde.icbc.DirichletBC(geom, lambda x: val_psi_anode, on_anode, component=0)
        bc_phin_a = dde.icbc.DirichletBC(geom, lambda x: val_phi_anode, on_anode, component=1)
        bc_phip_a = dde.icbc.DirichletBC(geom, lambda x: val_phi_anode, on_anode, component=2)
        
        return [bc_psi_c, bc_phin_c, bc_phip_c, bc_psi_a, bc_phin_a, bc_phip_a]

# --- 5. ANALYTICAL GUESS (For Pre-training) ---
def analytical_guess(x_numpy):
    y = x_numpy[:, 1:2]
    
    psi_n = PhysicsConstants.V_t * np.log(DeviceGeometry.N_D_cathode / PhysicsConstants.n_i)
    psi_p = -PhysicsConstants.V_t * np.log(DeviceGeometry.N_A_bulk / PhysicsConstants.n_i)
    psi_pp = -PhysicsConstants.V_t * np.log(DeviceGeometry.N_A_anode / PhysicsConstants.n_i)
    
    k = 20.0
    sig_1 = 0.5 * (1 + np.tanh(k * (y - DeviceGeometry.y_j1)))
    sig_2 = 0.5 * (1 + np.tanh(k * (y - DeviceGeometry.y_j2)))
    
    psi_guess = psi_n * (1 - sig_1) + psi_p * (sig_1 * (1 - sig_2)) + psi_pp * sig_2
    
    phi_n_guess = np.zeros_like(y)
    phi_p_guess = np.zeros_like(y)
    
    return np.hstack((psi_guess, phi_n_guess, phi_p_guess))

# --- 6. MODEL FACTORY ---
def create_model(V_bias, optical_params=None, initial_weights=None):
    physics = PhotodiodePhysics(V_bias, optical_params)
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
        geom,
        physics.pde,
        bcs,
        num_domain=0, 
        num_boundary=400,
        anchors=junction_sampler(3000)
    )
    
    def feature_transform(x):
        x_norm = torch.zeros_like(x)
        x_norm[:, 0] = (x[:, 0] - DeviceGeometry.width_um/2) / (DeviceGeometry.width_um/2)
        x_norm[:, 1] = (x[:, 1] - DeviceGeometry.y_total_um/2) / (DeviceGeometry.y_total_um/2)
        return x_norm

    net = dde.nn.FNN([2] + [64] * 6 + [3], "tanh", "Glorot normal")
    net.apply_feature_transform(feature_transform)
    
    model = dde.Model(data, net)
    
    if initial_weights:
        model.net.load_state_dict(initial_weights)
        logger.info("Loaded weights from previous state.")
    
    return model

# --- 7. UTILITIES: PRE-TRAINING ---
def pretrain_analytical(model, iters=5000):
    logger.info(">>> Pre-training on Analytical Guess...")
    geom = model.data.geom
    X_train = geom.random_points(2000)
    Y_train = analytical_guess(X_train)
    
    X_tensor = torch.from_numpy(X_train).float().to(device)
    Y_tensor = torch.from_numpy(Y_train).float().to(device)
    
    optimizer = torch.optim.Adam(model.net.parameters(), lr=1e-3)
    model.net.train()
    
    for i in range(iters):
        optimizer.zero_grad()
        y_pred = model.net(X_tensor)
        loss = torch.mean(torch.square(y_pred - Y_tensor))
        loss.backward()
        optimizer.step()
        
        if i % 1000 == 0:
            logger.info(f"   Step {i}: Fitting Loss {loss.item():.2e}")
    logger.info(">>> Pre-training complete.")

# --- 8. POST-PROCESSING ---
def calculate_current(model):
    n_pts = 500
    x_vals_um = np.linspace(0, DeviceGeometry.width_um, n_pts)
    
    # CORRECTION: Offset by 0.1% of device height to safely avoid BC singularities
    eps = 0.001 * DeviceGeometry.y_total_um
    y_vals_um = np.full_like(x_vals_um, DeviceGeometry.y_total_um - eps)
    coords = np.column_stack((x_vals_um, y_vals_um))
    
    try:
        x_tensor = torch.tensor(coords, dtype=torch.float32, device=device)
        x_tensor.requires_grad = True
        
        u = model.net(x_tensor)
        psi = u[:, 0:1]
        phi_n = u[:, 1:2]
        phi_p = u[:, 2:3]
        
        limit = 30.0
        n = PhysicsConstants.n_i * torch.exp(torch.clamp((psi - phi_n) / PhysicsConstants.V_t, -limit, limit))
        p = PhysicsConstants.n_i * torch.exp(torch.clamp((phi_p - psi) / PhysicsConstants.V_t, -limit, limit))
        
        grad_phin = torch.autograd.grad(phi_n, x_tensor, torch.ones_like(phi_n), create_graph=False)[0]
        grad_phip = torch.autograd.grad(phi_p, x_tensor, torch.ones_like(phi_p), create_graph=False)[0]
        
        dphin_dy_cm = grad_phin[:, 1:2] * (1.0 / Scaling.L_scale)
        dphip_dy_cm = grad_phip[:, 1:2] * (1.0 / Scaling.L_scale)
        
        Jn_y = -PhysicsConstants.q * PhysicsConstants.mu_n * n * dphin_dy_cm
        Jp_y = -PhysicsConstants.q * PhysicsConstants.mu_p * p * dphip_dy_cm
        J_total = Jn_y + Jp_y
        
        J_vals = J_total.cpu().detach().numpy().flatten()
        
        del x_tensor, u, n, p, grad_phin, grad_phip
        # Cleanup CPU memory too
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        x_vals_cm = x_vals_um * Scaling.L_scale
        current_per_cm_depth = spi.trapezoid(J_vals, x_vals_cm)
        
        return current_per_cm_depth

    except RuntimeError as e:
        logger.error(f"Autograd/Runtime error in current calculation: {e}")
        return float('nan')
    except Exception as e:
        logger.error(f"General error in current calculation: {e}")
        return float('nan')

def plot_results(model, bias, label):
    x = np.linspace(0, DeviceGeometry.width_um, 100)
    y = np.linspace(0, DeviceGeometry.y_total_um, 200)
    X, Y = np.meshgrid(x, y)
    pts = np.vstack((X.ravel(), Y.ravel())).T
    
    u = model.predict(pts)
    psi = u[:, 0].reshape(X.shape)
    
    fig, ax = plt.subplots(figsize=(6, 5))
    c = ax.contourf(X, Y, psi, levels=50, cmap='viridis')
    plt.colorbar(c, label='Potential (V)')
    ax.set_title(f'Potential Profile ({label}, {bias}V)')
    ax.set_xlabel('Width (um)')
    ax.set_ylabel('Depth (um)')
    ax.invert_yaxis()
    save_plot(fig, f"potential_{label}_{bias}V.png")

# --- 9. MAIN SIMULATION ---
def run_simulation():
    setup_directories()
    dark_weights_db = {} 
    voltages = [0.0, -0.5, -1.0]
    
    logger.info("=== STEP 1: DARK CURRENT SIMULATION ===")
    dark_currents = []
    last_weights = None
    
    for i, v in enumerate(voltages):
        logger.info(f"\n--- Solving Dark Condition: V_bias = {v} V ---")
        
        model = create_model(v, optical_params=None, initial_weights=last_weights)
        
        if i == 0:
            pretrain_analytical(model, iters=5000)
        
        # Weights: [Poisson, Elec, Hole, BCs...]
        loss_weights = [1.0, 1.0, 1.0] + [50.0]*6 
        
        model.compile("adam", lr=1e-3, loss_weights=loss_weights)
        model.train(iterations=10000, display_every=1000)
        
        # Increase fine-tuning iterations for non-zero bias
        fine_tune_iters = 5000 if v == 0.0 else 8000
        model.compile("adam", lr=1e-4, loss_weights=loss_weights)
        model.train(iterations=fine_tune_iters, display_every=1000)
        
        I_dark = calculate_current(model)
        dark_currents.append(I_dark)
        logger.info(f"  >>> Dark Current: {I_dark:.4e} A/cm")
        
        plot_results(model, v, "Dark")
        
        current_weights = copy.deepcopy(model.net.state_dict())
        dark_weights_db[v] = current_weights
        last_weights = current_weights

    logger.info("=== STEP 2: ILLUMINATED SIMULATION & EQE ===")
    
    wavelength_nm = 850
    # Validation
    assert 300 < wavelength_nm < 1200, "Wavelength outside Si bandgap range"
    
    P_opt_W_cm2 = 0.1 
    assert P_opt_W_cm2 > 0, "Optical power must be positive"
    
    E_ph = (6.626e-34 * 3e8) / (wavelength_nm * 1e-9)
    Phi_0 = P_opt_W_cm2 / E_ph 
    
    alpha = 600.0 # cm^-1
    # Include surface reflection
    G0_eff = Phi_0 * (1 - PhysicsConstants.R_reflectance) * alpha
    
    optical_params = {
        'G0': G0_eff,
        'alpha': alpha
    }
    
    photo_currents = []
    eqes = []
    
    for i, v in enumerate(voltages):
        logger.info(f"\n--- Solving Light Condition: V_bias = {v} V ---")
        
        start_weights = dark_weights_db[v]
        model = create_model(v, optical_params=optical_params, initial_weights=start_weights)
        
        loss_weights = [1.0, 1.0, 1.0] + [50.0]*6 
        model.compile("adam", lr=1e-4, loss_weights=loss_weights)
        
        # CORRECTION: More iterations for light case to handle generation term shock
        model.train(iterations=10000, display_every=1000)
        
        I_light = calculate_current(model)
        I_photo = abs(I_light - dark_currents[i])
        photo_currents.append(I_photo)
        
        device_width_cm = DeviceGeometry.width_um * Scaling.L_scale
        photons_in_per_sec = Phi_0 * device_width_cm
        electrons_out_per_sec = I_photo / PhysicsConstants.q
        
        eqe = (electrons_out_per_sec / photons_in_per_sec) * 100
        eqes.append(eqe)
        
        logger.info(f"  >>> I_light: {I_light:.4e} | I_photo: {I_photo:.4e}")
        logger.info(f"  >>> EQE: {eqe:.2f}%")
        
        plot_results(model, v, "Light")

    plt.figure()
    plt.plot(voltages, eqes, 'o-')
    plt.xlabel("Reverse Bias (V)")
    plt.ylabel("EQE (%)")
    plt.title(f"Quantum Efficiency at {wavelength_nm}nm")
    plt.grid(True)
    save_plot(plt.gcf(), "EQE_Curve.png")
    logger.info("Simulation Complete.")

if __name__ == "__main__":
    run_simulation()