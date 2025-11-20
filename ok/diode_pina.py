"""
PINA-based 2D P-N Photodiode Simulation - CORRECTED VERSION

This version properly uses the PINA library framework for:
- Physics-Informed Neural Network training
- Proper problem definition with conditions
- Domain handling
- Solver integration

This corrected version fixes:
1.  Adds the missing `calculate_current` method to SimplePhotodiodeSolver.
2.  Resolves unit ambiguity: calculations, printouts, and plots now
    consistently refer to Current Density (A/cm²).
3.  Corrects the physics formulas for EQE and Responsivity.
4.  Removes unused PINA imports (PINN, Trainer) as a custom
    solver loop is implemented.
5.  FIXED: Changed `func=torch.nn.Tanh()` to `func=torch.nn.Tanh` in the
    FeedForward model creation to pass the class, not an instance.

Requirements:
    pip install pina-mathlab torch matplotlib numpy
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pina import LabelTensor, Condition
from pina.problem import SpatialProblem
from pina.operator import grad, div
from pina.model import FeedForward
# from pina.solver import PINN # Not used in this custom loop
# from pina.trainer import Trainer # Not used in this custom loop
import os

# ======================================================================
# 1. PHYSICAL CONSTANTS
# ======================================================================
q = 1.602e-19  # Elementary charge [C]
k_B = 1.381e-23  # Boltzmann constant [J/K]
eps_0 = 8.854e-14  # Vacuum permittivity [F/cm]
T = 300.0  # Temperature [K]
V_t = k_B * T / q  # Thermal voltage ~0.0259 V
h = 6.626e-34  # Planck constant [J·s]
c = 2.998e10  # Speed of light [cm/s]

eps_r = 11.7  # Silicon relative permittivity
epsilon = eps_r * eps_0
n_i = 1.0e10  # Intrinsic carrier concentration [cm^-3]
n_i_sq = n_i ** 2

mu_n = 1400.0  # Electron mobility [cm^2/V/s]
mu_p = 450.0  # Hole mobility [cm^2/V/s]
D_n = mu_n * V_t  # Electron diffusivity [cm^2/s]
D_p = mu_p * V_t  # Hole diffusivity [cm^2/s]

tau_n = 1.0e-6  # Electron SRH lifetime [s]
tau_p = 1.0e-6  # Hole SRH lifetime [s]

# ======================================================================
# 2. GEOMETRY & DOPING
# ======================================================================
width_um = 100.0
n_plus_thickness_um = 0.5
p_thickness_um = 50.0
p_plus_thickness_um = 10.0

# Convert to cm
width_cm = width_um * 1e-4
n_plus_thickness = n_plus_thickness_um * 1e-4
p_thickness = p_thickness_um * 1e-4
p_plus_thickness = p_plus_thickness_um * 1e-4
total_depth = n_plus_thickness + p_thickness + p_plus_thickness

y_j1_cm = n_plus_thickness
y_j2_cm = n_plus_thickness + p_thickness

N_D_nplus = 1e19  # n+ doping [cm^-3]
N_A_p = 1e16  # p doping [cm^-3]
N_A_pplus = 1e19  # p+ doping [cm^-3]

# Normalization
scale_recomb = N_D_nplus / tau_n


def doping_profile(y):
    """Net doping N_D - N_A"""
    if not isinstance(y, torch.Tensor):
        y = torch.tensor(y, dtype=torch.float32)

    N_D = torch.where(y <= y_j1_cm,
                      torch.full_like(y, N_D_nplus),
                      torch.full_like(y, 0.0))

    N_A_val = torch.where(y <= y_j2_cm,
                          torch.full_like(y, N_A_p),
                          torch.full_like(y, N_A_pplus))
    N_A = torch.where(y > y_j1_cm, N_A_val, torch.full_like(y, 0.0))

    return (N_D - N_A).reshape(-1, 1)


# ======================================================================
# 3. OPTICAL GENERATION MODEL
# ======================================================================

class OpticalGeneration:
    """Optical generation rate calculator"""

    @staticmethod
    def absorption_coefficient(wavelength_nm):
        """Silicon absorption coefficient α(λ) [cm^-1]"""
        if wavelength_nm < 400:
            return 1e5
        elif wavelength_nm < 600:
            return 1e4
        elif wavelength_nm < 800:
            return 1e3
        elif wavelength_nm < 1000:
            return 100
        else:
            return 10

    @staticmethod
    def reflectance(wavelength_nm):
        """Surface reflectance"""
        return 0.3

    @staticmethod
    def photon_flux(power_density_mW_cm2, wavelength_nm):
        """Convert optical power density to photon flux density"""
        wavelength_cm = wavelength_nm * 1e-7
        power_W_cm2 = power_density_mW_cm2 * 1e-3
        photon_energy_J = h * c / wavelength_cm
        # Returns photon flux density [photons/s/cm^2]
        return power_W_cm2 / photon_energy_J

    @staticmethod
    def generation_rate(y, wavelength_nm, power_density_mW_cm2):
        """Calculate G(y, λ) [cm^-3·s^-1]"""
        alpha = OpticalGeneration.absorption_coefficient(wavelength_nm)
        R = OpticalGeneration.reflectance(wavelength_nm)
        Phi_0 = OpticalGeneration.photon_flux(power_density_mW_cm2, wavelength_nm)

        if isinstance(y, torch.Tensor):
            y_np = y.detach().cpu().numpy()
            G_np = alpha * (1 - R) * Phi_0 * np.exp(-alpha * y_np)
            return torch.tensor(G_np, dtype=torch.float32, device=y.device)
        else:
            return alpha * (1 - R) * Phi_0 * np.exp(-alpha * y)


# ======================================================================
# 4. PINA PROBLEM DEFINITION
# ======================================================================

class PhotodiodeProblem(SpatialProblem):
    """
    PINA Problem definition for photodiode simulation.

    Solves coupled drift-diffusion-Poisson equations:
    - Poisson: ∇²ψ = (q/ε)(n - p - N_net)
    - Electron continuity: ∇·J_n = q(G - U)
    - Hole continuity: ∇·J_p = -q(G - U)

    Output variables: psi (potential), log_n, log_p
    """

    output_variables = ['psi', 'log_n', 'log_p']
    spatial_domain = {'x': [0, width_cm], 'y': [0, total_depth]}

    def __init__(self, V_bias=0.0, G_func=None):
        """
        Initialize photodiode problem.

        Args:
            V_bias: Applied bias voltage [V]
            G_func: Optional generation function G(y)
        """
        super().__init__()
        self.V_bias = V_bias
        self.G_func = G_func

        # Define domain bounds
        x_min, x_max = 0.0, width_cm
        y_min, y_max = 0.0, total_depth

        # Create spatial domain
        self.domain = {'x': [x_min, x_max], 'y': [y_min, y_max]}

        # Define conditions
        self.conditions = {
            'D': Condition(location=self._sample_domain),  # Bulk domain
            'cathode': Condition(location=self._sample_cathode),  # Top contact
            'anode': Condition(location=self._sample_anode),  # Bottom contact
        }

    def _sample_domain(self, n_points):
        """Sample points in bulk domain"""
        pts = torch.rand(n_points, 2)
        pts[:, 0] = pts[:, 0] * width_cm  # x
        pts[:, 1] = pts[:, 1] * total_depth  # y
        return LabelTensor(pts, ['x', 'y'])

    def _sample_cathode(self, n_points):
        """Sample points on cathode (y=0)"""
        pts = torch.zeros(n_points, 2)
        pts[:, 0] = torch.rand(n_points) * width_cm  # x
        pts[:, 1] = 0.0  # y = 0
        return LabelTensor(pts, ['x', 'y'])

    def _sample_anode(self, n_points):
        """Sample points on anode (y=total_depth)"""
        pts = torch.zeros(n_points, 2)
        pts[:, 0] = torch.rand(n_points) * width_cm  # x
        pts[:, 1] = total_depth  # y = total_depth
        return LabelTensor(pts, ['x', 'y'])

    def poisson_equation(self, input_, output_):
        """Poisson's equation: ε∇²ψ = q(p - n + N_net)"""
        psi = output_.extract(['psi'])
        log_n = output_.extract(['log_n'])
        log_p = output_.extract(['log_p'])

        # Clamp for stability
        log_n_clamped = torch.clamp(log_n, min=np.log(1e5), max=np.log(1e20))
        log_p_clamped = torch.clamp(log_p, min=np.log(1e5), max=np.log(1e20))

        n = torch.exp(log_n_clamped)
        p = torch.exp(log_p_clamped)

        # Laplacian of psi
        laplacian_psi = div(grad(psi, input_), input_)

        # Net doping
        y = input_.extract(['y'])
        N_net = doping_profile(y)

        # Poisson residual
        rho = q * (p - n + N_net)
        residual = epsilon * laplacian_psi - rho

        # Normalize
        return residual / (q * N_A_pplus)

    def electron_continuity(self, input_, output_):
        """Electron continuity equation"""
        psi = output_.extract(['psi'])
        log_n = output_.extract(['log_n'])
        log_p = output_.extract(['log_p'])

        log_n_clamped = torch.clamp(log_n, min=np.log(1e5), max=np.log(1e20))
        log_p_clamped = torch.clamp(log_p, min=np.log(1e5), max=np.log(1e20))

        n = torch.exp(log_n_clamped)
        p = torch.exp(log_p_clamped)

        # Current density: J_n = -q·μ_n·n·∇ψ + q·D_n·∇n
        grad_psi = grad(psi, input_)
        grad_n = grad(n, input_)

        J_n = -q * mu_n * n * grad_psi + q * D_n * grad_n

        # Divergence of J_n
        div_Jn = div(J_n, input_)

        # Recombination (SRH)
        U_num = n * p - n_i_sq
        U_den = tau_p * (n + n_i) + tau_n * (p + n_i)
        U = U_num / (U_den + 1e-10)

        # Generation
        if self.G_func is None:
            G = torch.zeros_like(U)
        else:
            y = input_.extract(['y'])
            G = self.G_func(y)

        # Continuity equation
        residual = (div_Jn / q) - U + G

        return residual / scale_recomb

    def hole_continuity(self, input_, output_):
        """Hole continuity equation"""
        psi = output_.extract(['psi'])
        log_n = output_.extract(['log_n'])
        log_p = output_.extract(['log_p'])

        log_n_clamped = torch.clamp(log_n, min=np.log(1e5), max=np.log(1e20))
        log_p_clamped = torch.clamp(log_p, min=np.log(1e5), max=np.log(1e20))

        n = torch.exp(log_n_clamped)
        p = torch.exp(log_p_clamped)

        # Current density: J_p = -q·μ_p·p·∇ψ - q·D_p·∇p
        grad_psi = grad(psi, input_)
        grad_p = grad(p, input_)

        J_p = -q * mu_p * p * grad_psi - q * D_p * grad_p

        # Divergence of J_p
        div_Jp = div(J_p, input_)

        # Recombination (SRH)
        U_num = n * p - n_i_sq
        U_den = tau_p * (n + n_i) + tau_n * (p + n_i)
        U = U_num / (U_den + 1e-10)

        # Generation
        if self.G_func is None:
            G = torch.zeros_like(U)
        else:
            y = input_.extract(['y'])
            G = self.G_func(y)

        # Continuity equation
        residual = -(div_Jp / q) - U + G

        return residual / scale_recomb

    def cathode_bc(self, input_, output_):
        """Boundary condition at cathode"""
        psi = output_.extract(['psi'])
        log_n = output_.extract(['log_n'])
        log_p = output_.extract(['log_p'])

        V_bi_cathode = V_t * np.log(N_D_nplus / n_i)
        psi_target = V_bi_cathode + self.V_bias
        log_n_target = np.log(N_D_nplus)
        log_p_target = np.log(n_i_sq / N_D_nplus)

        return torch.cat([
            psi - psi_target,
            log_n - log_n_target,
            log_p - log_p_target
        ], dim=1)

    def anode_bc(self, input_, output_):
        """Boundary condition at anode"""
        psi = output_.extract(['psi'])
        log_n = output_.extract(['log_n'])
        log_p = output_.extract(['log_p'])

        V_bi_anode = -V_t * np.log(N_A_pplus / n_i)
        psi_target = V_bi_anode
        log_n_target = np.log(n_i_sq / N_A_pplus)
        log_p_target = np.log(N_A_pplus)

        return torch.cat([
            psi - psi_target,
            log_n - log_n_target,
            log_p - log_p_target
        ], dim=1)

    # Define the truth functions for PINA
    truth_domain = None  # PDE residuals should be zero
    truth_cathode = torch.zeros  # BC residuals should be zero
    truth_anode = torch.zeros


# ======================================================================
# 5. SIMPLIFIED TRAINING AND ANALYSIS (without full PINA Trainer)
# ======================================================================

class SimplePhotodiodeSolver:
    """
    Simplified solver that uses PINA concepts but with custom training loop.
    This is more practical for complex physics problems.
    """

    def __init__(self, device='cpu'):
        self.device = device
        self.model = None
        self.problem = None

    def create_model(self, layers=[64, 64, 64, 64]):
        """Create neural network model"""

        # ======================================================================
        # ============= START: CORRECTED LINE ================================
        # ======================================================================
        # We pass the class torch.nn.Tanh, not an instance torch.nn.Tanh()
        # The PINA FeedForward constructor will create instances itself.
        self.model = FeedForward(
            input_dimensions=2,
            output_dimensions=3,
            layers=layers,
            func=torch.nn.Tanh
        ).to(self.device)
        # ======================================================================
        # ============= END: CORRECTED LINE ==================================
        # ======================================================================

        # Initialize weights
        for m in self.model.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_normal_(m.weight)
                torch.nn.init.zeros_(m.bias)

        return self.model

    def setup_problem(self, V_bias=0.0, G_func=None):
        """Setup physics problem"""
        self.problem = PhotodiodeProblem(V_bias=V_bias, G_func=G_func)
        return self.problem

    def train_equilibrium(self, n_epochs=10000, lr=1e-3, n_domain=2000, n_bc=500):
        """Train for equilibrium conditions"""
        if self.model is None or self.problem is None:
            raise ValueError("Must create model and setup problem first")

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        print("\n" + "=" * 70)
        print("TRAINING: EQUILIBRIUM (V=0, Dark)")
        print("=" * 70)

        losses_history = []

        for epoch in range(n_epochs):
            optimizer.zero_grad()

            # Sample points
            pts_domain = self.problem.conditions['D'].location(n_domain).to(self.device)
            pts_cathode = self.problem.conditions['cathode'].location(n_bc).to(self.device)
            pts_anode = self.problem.conditions['anode'].location(n_bc).to(self.device)

            # Forward pass - domain
            out_domain = self.model(pts_domain)
            out_domain_labeled = LabelTensor(out_domain, self.problem.output_variables)

            # Compute PDE residuals
            res_poisson = self.problem.poisson_equation(pts_domain, out_domain_labeled)
            res_electron = self.problem.electron_continuity(pts_domain, out_domain_labeled)
            res_hole = self.problem.hole_continuity(pts_domain, out_domain_labeled)

            loss_pde = (res_poisson ** 2).mean() + \
                       (res_electron ** 2).mean() + \
                       (res_hole ** 2).mean()

            # Forward pass - boundaries
            out_cathode = self.model(pts_cathode)
            out_cathode_labeled = LabelTensor(out_cathode, self.problem.output_variables)
            bc_cathode = self.problem.cathode_bc(pts_cathode, out_cathode_labeled)

            out_anode = self.model(pts_anode)
            out_anode_labeled = LabelTensor(out_anode, self.problem.output_variables)
            bc_anode = self.problem.anode_bc(pts_anode, out_anode_labeled)

            loss_bc = (bc_cathode ** 2).mean() + (bc_anode ** 2).mean()

            # Total loss
            loss = 0.1 * loss_pde + 100.0 * loss_bc

            loss.backward()
            optimizer.step()

            losses_history.append(loss.item())

            if (epoch + 1) % 1000 == 0:
                print(f"Epoch {epoch + 1}/{n_epochs} | Loss: {loss.item():.6e} | "
                      f"PDE: {loss_pde.item():.6e} | BC: {loss_bc.item():.6e}")

        print("✓ Equilibrium training complete!")
        return losses_history

    def calculate_current(self, V_bias, G_func):
        """
        Calculates the terminal current density (J_y) [A/cm²].

        Since the physics is 1D (all variation is in y), the current
        density J_y = J_n_y + J_p_y must be constant wrt y in steady state
        (ignoring G/R). We calculate it along a line in y and average.

        Args:
            V_bias (float): The bias voltage (unused, taken from self.problem)
            G_func (callable): The generation function (unused, taken from self.problem)

        Returns:
            float: The calculated terminal current density J_y in [A/cm²].
        """
        self.model.eval()

        # Create a line of points along y at the center x
        ny = 200
        y_line = torch.linspace(0, total_depth, ny, device=self.device).reshape(-1, 1)
        x_line = torch.full_like(y_line, width_cm / 2)

        pts = torch.cat([x_line, y_line], dim=1)
        pts_labeled = LabelTensor(pts, ['x', 'y']).to(self.device)
        pts_labeled.requires_grad_(True)

        with torch.no_grad():
            output = self.model(pts_labeled)
            psi = output.extract(['psi'])

            log_n = output.extract(['log_n'])
            log_p = output.extract(['log_p'])

            log_n_clamped = torch.clamp(log_n, min=np.log(1e5), max=np.log(1e20))
            log_p_clamped = torch.clamp(log_p, min=np.log(1e5), max=np.log(1e20))

            n = torch.exp(log_n_clamped)
            p = torch.exp(log_p_clamped)

        # Calculate gradients
        grad_psi = grad(psi, pts_labeled, allow_unused=True)
        grad_n = grad(n, pts_labeled, allow_unused=True)
        grad_p = grad(p, pts_labeled, allow_unused=True)

        # Extract y-components of gradients
        dpsi_dy = grad_psi.extract(['y'])
        dn_dy = grad_n.extract(['y'])
        dp_dy = grad_p.extract(['y'])

        # Calculate y-components of current densities
        # J_n_y = -q·μ_n·n·(dψ/dy) + q·D_n·(dn/dy)
        # J_p_y = -q·μ_p·p·(dψ/dy) - q·D_p·(dp/dy)

        J_n_y = -q * mu_n * n * dpsi_dy + q * D_n * dn_dy
        J_p_y = -q * mu_p * p * dpsi_dy - q * D_p * dp_dy

        J_total_y = J_n_y + J_p_y

        # In steady state, J_total_y should be constant.
        # We average over the line to get a robust estimate.
        return J_total_y.mean().item()

    def generate_iv_curve(self, voltage_range, G_func=None, n_adapt_epochs=1000, lr=1e-4):
        """Generate I-V curve by adapting model for each voltage point"""
        voltages = []
        currents_density = []

        print("\n" + "=" * 70)
        curve_type = "ILLUMINATED" if G_func is not None else "DARK"
        print(f"GENERATING {curve_type} I-V CURVE")
        print("=" * 70)

        for V in voltage_range:
            print(f"\nBias voltage: {V:.3f} V")

            # Update problem with new bias
            self.problem.V_bias = V
            self.problem.G_func = G_func

            # Quick adaptation training
            optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

            for epoch in range(n_adapt_epochs):
                optimizer.zero_grad()

                pts_domain = self.problem.conditions['D'].location(1000).to(self.device)
                pts_cathode = self.problem.conditions['cathode'].location(200).to(self.device)
                pts_anode = self.problem.conditions['anode'].location(200).to(self.device)

                # PDE residuals
                out_domain = self.model(pts_domain)
                out_domain_labeled = LabelTensor(out_domain, self.problem.output_variables)

                res_poisson = self.problem.poisson_equation(pts_domain, out_domain_labeled)
                res_electron = self.problem.electron_continuity(pts_domain, out_domain_labeled)
                res_hole = self.problem.hole_continuity(pts_domain, out_domain_labeled)

                loss_pde = (res_poisson ** 2).mean() + \
                           (res_electron ** 2).mean() + \
                           (res_hole ** 2).mean()

                # BC residuals
                out_cathode = self.model(pts_cathode)
                out_cathode_labeled = LabelTensor(out_cathode, self.problem.output_variables)
                bc_cathode = self.problem.cathode_bc(pts_cathode, out_cathode_labeled)

                out_anode = self.model(pts_anode)
                out_anode_labeled = LabelTensor(out_anode, self.problem.output_variables)
                bc_anode = self.problem.anode_bc(pts_anode, out_anode_labeled)

                loss_bc = (bc_cathode ** 2).mean() + (bc_anode ** 2).mean()

                loss = 0.1 * loss_pde + 100.0 * loss_bc
                loss.backward()
                optimizer.step()

                if (epoch + 1) % 200 == 0:
                    print(f"  Adaptation epoch {epoch + 1}/{n_adapt_epochs} | Loss: {loss.item():.6e}")

            # Calculate current density
            J = self.calculate_current(V, G_func)
            voltages.append(V)
            currents_density.append(J)
            print(f"  → Current Density: {J:.6e} A/cm²")

        print(f"\n✓ {curve_type} I-V curve complete!")
        return np.array(voltages), np.array(currents_density)

    def calculate_quantum_efficiency(self, wavelengths_nm, power_density_mW_cm2=1.0, V_bias=0.0):
        """Calculate external quantum efficiency vs wavelength"""
        print("\n" + "=" * 70)
        print("CALCULATING QUANTUM EFFICIENCY")
        print("=" * 70)

        wavelengths = []
        EQE_values = []
        responsivity_values = []

        # Calculate dark current density
        print(f"\nCalculating dark current density at V={V_bias:.3f} V...")
        J_dark = self.calculate_current(V_bias, G_func=None)
        print(f"Dark current density: {J_dark:.6e} A/cm²")

        for wl in wavelengths_nm:
            print(f"\nWavelength: {wl} nm")

            # Define generation function for this wavelength
            def G_func(y):
                return OpticalGeneration.generation_rate(y, wl, power_density_mW_cm2)

            # Train with illumination
            print("  Training with illumination...")
            self.problem.V_bias = V_bias
            self.problem.G_func = G_func

            optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)

            for epoch in range(500):
                optimizer.zero_grad()

                pts_domain = self.problem.conditions['D'].location(1000).to(self.device)
                pts_cathode = self.problem.conditions['cathode'].location(200).to(self.device)
                pts_anode = self.problem.conditions['anode'].location(200).to(self.device)

                out_domain = self.model(pts_domain)
                out_domain_labeled = LabelTensor(out_domain, self.problem.output_variables)

                res_poisson = self.problem.poisson_equation(pts_domain, out_domain_labeled)
                res_electron = self.problem.electron_continuity(pts_domain, out_domain_labeled)
                res_hole = self.problem.hole_continuity(pts_domain, out_domain_labeled)

                loss_pde = (res_poisson ** 2).mean() + \
                           (res_electron ** 2).mean() + \
                           (res_hole ** 2).mean()

                out_cathode = self.model(pts_cathode)
                out_cathode_labeled = LabelTensor(out_cathode, self.problem.output_variables)
                bc_cathode = self.problem.cathode_bc(pts_cathode, out_cathode_labeled)

                out_anode = self.model(pts_anode)
                out_anode_labeled = LabelTensor(out_anode, self.problem.output_variables)
                bc_anode = self.problem.anode_bc(pts_anode, out_anode_labeled)

                loss_bc = (bc_cathode ** 2).mean() + (bc_anode ** 2).mean()

                loss = 0.1 * loss_pde + 100.0 * loss_bc
                loss.backward()
                optimizer.step()

            # Calculate illuminated current density
            J_light = self.calculate_current(V_bias, G_func)
            J_ph = J_light - J_dark  # Photocurrent density

            print(f"  Light current density: {J_light:.6e} A/cm²")
            print(f"  Photocurrent density: {J_ph:.6e} A/cm²")

            # Calculate incident photon flux density
            Phi_inc = OpticalGeneration.photon_flux(power_density_mW_cm2, wl)  # [photons/s/cm^2]
            power_density_W_cm2 = power_density_mW_cm2 * 1e-3  # [W/cm^2]

            # EQE = (electrons collected per sec per area) / (photons incident per sec per area)
            # EQE = (J_ph / q) / Phi_inc
            EQE = abs(J_ph) / (q * Phi_inc) if Phi_inc > 0 else 0

            # Responsivity [A/W] = J_ph [A/cm^2] / P_inc [W/cm^2]
            responsivity = abs(J_ph) / power_density_W_cm2 if power_density_W_cm2 > 0 else 0

            wavelengths.append(wl)
            EQE_values.append(EQE)
            responsivity_values.append(responsivity)

            print(f"  EQE: {EQE * 100:.2f}%")
            print(f"  Responsivity: {responsivity:.4f} A/W")

        print("\n✓ Quantum efficiency calculation complete!")
        return np.array(wavelengths), np.array(EQE_values), np.array(responsivity_values)


# ======================================================================
# 6. PLOTTING FUNCTIONS
# ======================================================================

def plot_training_loss(losses, filename='training_loss.png'):
    """Plot training loss history"""
    plt.figure(figsize=(10, 6))
    plt.semilogy(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss History')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"✓ Saved: {filename}")


def plot_2d_solution(model, device='cpu', filename='equilibrium_2d.png'):
    """Plot 2D contours of psi, n, p"""
    model.eval()

    # Create mesh grid
    nx, ny = 100, 100
    x = torch.linspace(0, width_cm, nx, device=device)
    y = torch.linspace(0, total_depth, ny, device=device)
    X, Y = torch.meshgrid(x, y, indexing='ij')

    x_flat = X.reshape(-1, 1)
    y_flat = Y.reshape(-1, 1)

    pts = torch.cat([x_flat, y_flat], dim=1)
    pts_labeled = LabelTensor(pts, ['x', 'y'])

    with torch.no_grad():
        output = model(pts_labeled)
        psi = output[:, 0:1]
        log_n = output[:, 1:2]
        log_p = output[:, 2:3]

        n = torch.exp(torch.clamp(log_n, min=np.log(1e5), max=np.log(1e20)))
        p = torch.exp(torch.clamp(log_p, min=np.log(1e5), max=np.log(1e20)))

    # Reshape for plotting
    psi_2d = psi.reshape(nx, ny).cpu().numpy()
    n_2d = n.reshape(nx, ny).cpu().numpy()
    p_2d = p.reshape(nx, ny).cpu().numpy()
    X_np = X.cpu().numpy() * 1e4  # Convert to μm
    Y_np = Y.cpu().numpy() * 1e4

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Potential
    im1 = axes[0].contourf(X_np, Y_np, psi_2d, levels=50, cmap='RdBu_r')
    axes[0].set_xlabel('x (μm)')
    axes[0].set_ylabel('y (μm)')
    axes[0].set_title('Electrostatic Potential ψ (V)')
    plt.colorbar(im1, ax=axes[0])

    # Electron concentration (log scale)
    im2 = axes[1].contourf(X_np, Y_np, np.log10(n_2d), levels=50, cmap='viridis')
    axes[1].set_xlabel('x (μm)')
    axes[1].set_ylabel('y (μm)')
    axes[1].set_title('Electron Concentration log₁₀(n) [cm⁻³]')
    plt.colorbar(im2, ax=axes[1])

    # Hole concentration (log scale)
    im3 = axes[2].contourf(X_np, Y_np, np.log10(p_2d), levels=50, cmap='plasma')
    axes[2].set_xlabel('x (μm)')
    axes[2].set_ylabel('y (μm)')
    axes[2].set_title('Hole Concentration log₁₀(p) [cm⁻³]')
    plt.colorbar(im3, ax=axes[2])

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"✓ Saved: {filename}")


def plot_1d_profile(model, x_position=None, device='cpu', filename='equilibrium_1d.png'):
    """Plot 1D profiles along y-axis at given x position"""
    model.eval()

    if x_position is None:
        x_position = width_cm / 2  # Center of device

    ny = 200
    y = torch.linspace(0, total_depth, ny, device=device).reshape(-1, 1)
    x = torch.ones_like(y) * x_position

    pts = torch.cat([x, y], dim=1)
    pts_labeled = LabelTensor(pts, ['x', 'y'])

    with torch.no_grad():
        output = model(pts_labeled)
        psi = output[:, 0:1]
        log_n = output[:, 1:2]
        log_p = output[:, 2:3]

        n = torch.exp(torch.clamp(log_n, min=np.log(1e5), max=np.log(1e20)))
        p = torch.exp(torch.clamp(log_p, min=np.log(1e5), max=np.log(1e20)))

    y_um = y.cpu().numpy().flatten() * 1e4
    psi_np = psi.cpu().numpy().flatten()
    n_np = n.cpu().numpy().flatten()
    p_np = p.cpu().numpy().flatten()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Potential
    axes[0].plot(y_um, psi_np, 'b-', linewidth=2)
    axes[0].axvline(y_j1_cm * 1e4, color='r', linestyle='--', alpha=0.5, label='n+/p junction')
    axes[0].axvline(y_j2_cm * 1e4, color='g', linestyle='--', alpha=0.5, label='p/p+ interface')
    axes[0].set_xlabel('Depth y (μm)')
    axes[0].set_ylabel('Potential ψ (V)')
    axes[0].set_title(f'Potential Profile at x={x_position * 1e4:.1f} μm')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Carrier concentrations (log scale)
    axes[1].semilogy(y_um, n_np, 'b-', linewidth=2, label='Electrons (n)')
    axes[1].semilogy(y_um, p_np, 'r-', linewidth=2, label='Holes (p)')
    axes[1].axvline(y_j1_cm * 1e4, color='k', linestyle='--', alpha=0.3)
    axes[1].axvline(y_j2_cm * 1e4, color='k', linestyle='--', alpha=0.3)
    axes[1].set_xlabel('Depth y (μm)')
    axes[1].set_ylabel('Carrier Concentration (cm⁻³)')
    axes[1].set_title('Carrier Concentration Profile')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Doping profile
    y_torch = torch.linspace(0, total_depth, ny, device=device).reshape(-1, 1)
    N_net = doping_profile(y_torch).cpu().numpy().flatten()

    axes[2].plot(y_um, N_net, 'k-', linewidth=2)
    axes[2].axhline(0, color='gray', linestyle='-', alpha=0.3)
    axes[2].axvline(y_j1_cm * 1e4, color='r', linestyle='--', alpha=0.5)
    axes[2].axvline(y_j2_cm * 1e4, color='g', linestyle='--', alpha=0.5)
    axes[2].set_xlabel('Depth y (μm)')
    axes[2].set_ylabel('Net Doping N_D - N_A (cm⁻³)')
    axes[2].set_title('Doping Profile')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"✓ Saved: {filename}")


def plot_iv_curves(V_dark, J_dark, V_light, J_light, filename='iv_curves.png'):
    """Plot dark and illuminated I-V curves"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Linear scale
    ax1.plot(V_dark, J_dark * 1e3, 'b-', linewidth=2, label='Dark')
    ax1.plot(V_light, J_light * 1e3, 'r-', linewidth=2, label='Illuminated')
    ax1.set_xlabel('Voltage (V)')
    ax1.set_ylabel('Current Density (mA/cm²)')
    ax1.set_title('I-V Characteristics')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Semi-log scale
    ax2.semilogy(V_dark, np.abs(J_dark) * 1e3, 'b-', linewidth=2, label='Dark')
    ax2.semilogy(V_light, np.abs(J_light) * 1e3, 'r-', linewidth=2, label='Illuminated')
    ax2.set_xlabel('Voltage (V)')
    ax2.set_ylabel('|Current Density| (mA/cm²)')
    ax2.set_title('I-V Characteristics (log scale)')
    ax2.legend()
    ax2.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"✓ Saved: {filename}")


def plot_quantum_efficiency(wavelengths, EQE, responsivity, filename='quantum_efficiency.png'):
    """Plot quantum efficiency and responsivity"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # EQE plot
    ax1.plot(wavelengths, EQE * 100, 'b-o', linewidth=2, markersize=8)
    ax1.set_xlabel('Wavelength (nm)')
    ax1.set_ylabel('External Quantum Efficiency (%)')
    ax1.set_title('Quantum Efficiency vs Wavelength')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([wavelengths.min() - 50, wavelengths.max() + 50])

    # Responsivity plot
    ax2.plot(wavelengths, responsivity, 'r-o', linewidth=2, markersize=8)
    ax2.set_xlabel('Wavelength (nm)')
    ax2.set_ylabel('Responsivity (A/W)')
    ax2.set_title('Responsivity vs Wavelength')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([wavelengths.min() - 50, wavelengths.max() + 50])

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"✓ Saved: {filename}")


# ======================================================================
# 7. MAIN EXECUTION
# ======================================================================

def main():
    """Main execution using PINA framework"""
    print("\n" + "=" * 70)
    print("2D PHOTODIODE SIMULATION - PINA FRAMEWORK")
    print("=" * 70)
    print("\nDevice Parameters:")
    print(f"  Width (x-dim): {width_um} μm")
    print(f"  n+ thickness: {n_plus_thickness_um} μm")
    print(f"  p thickness: {p_thickness_um} μm")
    print(f"  p+ thickness: {p_plus_thickness_um} μm")
    print(f"  Total depth (y-dim): {total_depth * 1e4:.1f} μm")
    print(f"\nDoping:")
    print(f"  N_D (n+): {N_D_nplus:.2e} cm⁻³")
    print(f"  N_A (p):  {N_A_p:.2e} cm⁻³")
    print(f"  N_A (p+): {N_A_pplus:.2e} cm⁻³")

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    # Create output directory
    os.makedirs('results', exist_ok=True)

    # ========== STEP 1: Train Equilibrium ==========
    print("\n" + "=" * 70)
    print("STEP 1: EQUILIBRIUM TRAINING")
    print("=" * 70)

    # Create solver
    solver = SimplePhotodiodeSolver(device=device)

    # Create model
    print("\nCreating neural network model...")
    solver.create_model(layers=[64, 64, 64, 64])

    # Setup problem
    print("Setting up physics problem...")
    solver.setup_problem(V_bias=0.0, G_func=None)

    # Train equilibrium
    losses = solver.train_equilibrium(n_epochs=10000, lr=1e-3)

    # Plot training loss
    plot_training_loss(losses, 'results/training_loss.png')

    # Plot 2D solution
    plot_2d_solution(solver.model, device, 'results/equilibrium_2d.png')

    # Plot 1D profiles
    plot_1d_profile(solver.model, device=device, filename='results/equilibrium_1d.png')

    # Save equilibrium model
    torch.save(solver.model.state_dict(), 'results/model_equilibrium.pt')

    # ========== STEP 2: Dark I-V Curve ==========
    print("\n" + "=" * 70)
    print("STEP 2: DARK I-V CURVE")
    print("=" * 70)

    # Generate dark I-V curve
    V_dark_range = np.linspace(-0.5, 0.6, 12)
    V_dark, J_dark = solver.generate_iv_curve(
        V_dark_range, G_func=None, n_adapt_epochs=1000, lr=1e-4
    )

    # Save data
    np.savetxt('results/dark_iv.txt', np.column_stack([V_dark, J_dark]),
               header='Voltage(V) Current_Density(A/cm^2)')

    # ========== STEP 3: Illuminated I-V Curve ==========
    print("\n" + "=" * 70)
    print("STEP 3: ILLUMINATED I-V CURVE")
    print("=" * 70)

    # Reload equilibrium model
    solver.model.load_state_dict(torch.load('results/model_equilibrium.pt'))

    # Define illumination (600 nm, 1 mW/cm²)
    wavelength_illumination = 600
    power_density = 1.0

    def G_func_illuminated(y):
        return OpticalGeneration.generation_rate(y, wavelength_illumination, power_density)

    print(f"Illumination: {wavelength_illumination} nm, {power_density} mW/cm²")

    V_light, J_light = solver.generate_iv_curve(
        V_dark_range, G_func=G_func_illuminated, n_adapt_epochs=1000, lr=1e-4
    )

    # Save data
    np.savetxt('results/illuminated_iv.txt', np.column_stack([V_light, J_light]),
               header='Voltage(V) Current_Density(A/cm^2)')

    # Plot I-V curves
    plot_iv_curves(V_dark, J_dark, V_light, J_light, 'results/iv_curves.png')

    # Calculate key parameters
    print("\n" + "-" * 70)
    print("KEY PARAMETERS:")
    print("-" * 70)

    # Open-circuit voltage
    idx_voc = np.argmin(np.abs(J_light))
    V_oc = V_light[idx_voc]
    print(f"Open-circuit voltage (Voc): {V_oc:.4f} V")

    # Short-circuit current density
    idx_jsc = np.argmin(np.abs(V_light))
    J_sc = J_light[idx_jsc]
    # Calculate total current assuming 1cm z-depth
    total_area = width_cm * 1.0  # [cm * cm] = cm^2
    I_sc = J_sc * total_area

    print(f"Short-circuit current density (Jsc): {J_sc:.6e} A/cm²")
    print(f"  Total short-circuit current (Isc): {I_sc:.6e} A (for {width_um}um width, 1cm depth)")

    # ========== STEP 4: Quantum Efficiency ==========
    print("\n" + "=" * 70)
    print("STEP 4: QUANTUM EFFICIENCY")
    print("=" * 70)

    # Reload equilibrium model
    solver.model.load_state_dict(torch.load('results/model_equilibrium.pt'))

    # Calculate QE at multiple wavelengths
    wavelengths_nm = np.array([400, 500, 600, 700, 800, 900, 1000])

    wl, EQE, responsivity = solver.calculate_quantum_efficiency(
        wavelengths_nm, power_density_mW_cm2=1.0, V_bias=0.0
    )

    # Save data
    np.savetxt('results/quantum_efficiency.txt',
               np.column_stack([wl, EQE * 100, responsivity]),
               header='Wavelength(nm) EQE(%) Responsivity(A/W)')

    # Plot QE
    plot_quantum_efficiency(wl, EQE, responsivity, 'results/quantum_efficiency.png')

    # ========== SUMMARY ==========
    print("\n" + "=" * 70)
    print("SIMULATION COMPLETE!")
    print("=" * 70)
    print("\nGenerated files in 'results/' directory:")
    print("  • training_loss.png           - Training convergence")
    print("  • equilibrium_2d.png          - 2D solution maps")
    print("  • equilibrium_1d.png          - 1D profiles")
    print("  • iv_curves.png               - Dark & illuminated I-V")
    print("  • quantum_efficiency.png      - QE & responsivity vs λ")
    print("  • dark_iv.txt                 - Dark I-V data")
    print("  • illuminated_iv.txt          - Illuminated I-V data")
    print("  • quantum_efficiency.txt      - QE data")
    print("  • model_equilibrium.pt        - Trained model")
    print("\nKey Results:")
    print(f"  V_oc = {V_oc:.4f} V")
    print(f"  J_sc = {J_sc:.6e} A/cm²")
    print(f"  I_sc = {I_sc:.6e} A (total for {width_um}um x 1cm device)")
    print(f"  Peak EQE = {np.max(EQE) * 100:.2f}% at {wl[np.argmax(EQE)]:.0f} nm")
    print(f"  Peak Responsivity = {np.max(responsivity):.4f} A/W")
    print("\n" + "=" * 70)
    print("✓ All simulations and plots completed successfully!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()