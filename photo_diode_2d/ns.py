print("""
  ╔════════════════════════════════════════════════════════════════════╗
  ║ 💡 COMPLETE 2D SIMULATION WORKFLOW & PHYSICS GUIDE                  ║
  ╚════════════════════════════════════════════════════════════════════╝
    The following steps outline a robust strategy to solve the
    Drift-Diffusion equations for your 2D photodiode. All physics
    and boundary conditions are tailored to your gmsh_diode2d.py setup.

  ╔════════════════════════════════════════════════════════════════════╗
  ║ STEP 1: DEFINE VARIABLES, PHYSICS & EQUATIONS                      ║
  ╚════════════════════════════════════════════════════════════════════╝
    First, define the core components of the simulation for all regions.

    A. Primary Solution Variables (Node Models):
       - Potential: $\psi(x, y)$
       - Electrons: $n(x, y)$
       - Holes: $p(x, y)$

    B. Physics Models (Auxiliary Models):
       - Electric Field Vector: $\mathbf{E} = -\nabla\psi = -\left( \frac{\partial \psi}{\partial x}\mathbf{\hat{x}} + \frac{\partial \psi}{\partial y}\mathbf{\hat{y}} \right)$
       - Current Density Vectors (Drift-Diffusion):
         $$ \mathbf{J}_n = q\mu_n n \mathbf{E} + q D_n \nabla n $$
         $$ \mathbf{J}_p = q\mu_p p \mathbf{E} - q D_p \nabla p $$
       - Recombination Rate Models ($U = U_{SRH} + U_{Auger} + U_{rad}$):
         • SRH: $U_{SRH} = \\frac{np - n_i^2}{\\tau_p(n + n_1) + \\tau_n(p + p_1)}$
         • Auger: $U_{Auger} = (C_n n + C_p p)(np - n_i^2)$
         • Radiative: $U_{rad} = B(np - n_i^2)$

    C. Governing Equations (2D Steady-State):
       - Poisson's Equation: $\nabla\cdot(\epsilon\nabla\psi) = -q(p - n + N_D^+ - N_A^-)$
       - Electron Continuity: $\frac{1}{q}\nabla\cdot \mathbf{J}_n = U - G$
       - Hole Continuity: $-\frac{1}{q}\nabla\cdot \mathbf{J}_p = U - G$

    D. Boundary & Interface Conditions:
       - Ohmic Contacts (Dirichlet BCs):
         • On `cathode`: $\psi = V_{applied}$, $n = N_D$, $p = n_i^2/N_D$
         • On `anode`: $\psi = 0$, $p = N_A$, $n = n_i^2/N_A$
       - Insulating Surfaces (Neumann BCs):
         • On `top_surface`, `left_side`, `right_side`:
         • No normal E-field: $\nabla \psi \cdot \mathbf{\hat{n}} = 0$
         • Surface Recombination: $\mathbf{J}_n \cdot \mathbf{\hat{n}} = q S_n (n - n_{eq})$

  ╔════════════════════════════════════════════════════════════════════╗
  ║ STEP 2: SOLVE FOR POTENTIAL ONLY (INITIAL GUESS)                   ║
  ╚════════════════════════════════════════════════════════════════════╝
    Solve only Poisson's equation first to establish the initial
    built-in potential across the junctions. This provides a stable
    starting point for the fully coupled solver.

  ╔════════════════════════════════════════════════════════════════════╗
  ║ STEP 3: SOLVE FULLY COUPLED SYSTEM IN DARK (EQUILIBRIUM)           ║
  ╚════════════════════════════════════════════════════════════════════╝
    Solve the complete system (Poisson, Electron Continuity, Hole
    Continuity) with no applied voltage ($V_{applied} = 0$) and no
    optical generation ($G = 0$) to find the thermal equilibrium state.

  ╔════════════════════════════════════════════════════════════════════╗
  ║ STEP 4: SIMULATE DARK I-V CHARACTERISTICS                          ║
  ╚════════════════════════════════════════════════════════════════════╝
    Sweep the applied voltage on the `cathode` to get the dark I-V curve.

    1. Loop through voltage steps (e.g., from -1V to 0.8V).
    2. At each step, solve the fully coupled system with $G=0$.
    3. Calculate Terminal Current by integrating the normal component of
       the total current density vector along a contact boundary:
       $$ I_{dark}(V) = \int_{\text{anode}} (\mathbf{J}_n + \mathbf{J}_p) \cdot \mathbf{\hat{n}} \, dl $$
       (Note: The result is in Amperes per meter [A/m] for a 2D simulation).

  ╔════════════════════════════════════════════════════════════════════╗
  ║ STEP 5: SIMULATE ILLUMINATED CHARACTERISTICS                       ║
  ╚════════════════════════════════════════════════════════════════════╝
    Introduce an optical generation term ($G > 0$) to simulate the
    photodiode's response to light.

    1. Define an optical generation model based on depth ($y$) from the
       illuminated surface ($y=0$):
       $$ G(y, \lambda) = \alpha(\lambda)[1 - R_s(\lambda)]\Phi_0(\lambda)e^{-\alpha(\lambda)y} $$
    2. Repeat the voltage sweep from Step 4 with the non-zero $G(y, \lambda)$.
    3. Extract the total illuminated current $I_{light}(V)$.
    4. Calculate Photocurrent: $I_{ph}(V) = I_{light}(V) - I_{dark}(V)$.
    5. Analyze key metrics: Short-circuit current ($I_{sc}$), Open-circuit
       voltage ($V_{oc}$), External Quantum Efficiency (EQE), and Responsivity ($\mathcal{R}$).

  ╔════════════════════════════════════════════════════════════════════╗
  ║ 📚 REFERENCE RESOURCES                                             ║
  ╚════════════════════════════════════════════════════════════════════╝
    • DEVSIM Documentation: https://devsim.org
    • Example scripts: devsim/examples/diode/
    • Physics reference: S.M. Sze, "Physics of Semiconductor Devices"
  """)