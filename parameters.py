# Silicon Material Parameters Dictionary
# All values are for Silicon at 300K unless otherwise noted
# References are provided for each parameter group
import numpy as np


SILICON_PARAMS = {
    # ==============================================================================
    # FUNDAMENTAL PHYSICAL CONSTANTS
    # ==============================================================================
    "permittivity": 11.7 * 8.854e-14,  # [F/cm] - Silicon relative permittivity × ε₀
    # Ref: Sze & Ng, Physics of Semiconductor Devices, 3rd Ed.

    "electron_charge": 1.602176634e-19,  # [C] - Elementary charge (exact value, 2019 SI definition)

    "bandgap": 1.124,  # [eV] - Silicon bandgap at 300K
    # Ref: Green, Solar Cells (1982), temperature-corrected value

    # ==============================================================================
    # EFFECTIVE DENSITY OF STATES
    # ==============================================================================
    "Nc_300K": 2.86e19,  # [cm⁻³] - Conduction band effective density of states at 300K
    # Ref: Green, J. Appl. Phys. 67, 2944 (1990)

    "Nv_300K": 3.10e19,  # [cm⁻³] - Valence band effective density of states at 300K
    # Ref: Green, J. Appl. Phys. 67, 2944 (1990)

    # ==============================================================================
    # DOPING PROFILE PARAMETERS (for p-n junction photodiode)
    # ==============================================================================
    "peak_p_doping": 1e18,  # [cm⁻³] - Peak acceptor concentration (Boron implant)
    # Typical for photodiode p+ region

    "doping_straggle": 0.1,  # [μm] - Gaussian straggle of implant profile (ΔRp)
    # Ref: Gibbons et al., Projected Range Statistics

    "n_bulk": 1e15,  # [cm⁻³] - N-type substrate doping (Phosphorus)
    # Typical for high-resistivity substrate

    "projected_range": 0.15,  # [μm] - Projected range (Rp) of p-type implant
    # Consistent with low-energy Boron implant

    # ==============================================================================
    # CARRIER LIFETIME PARAMETERS (Shockley-Read-Hall)
    # ==============================================================================
    # Based on Klaassen model for doping-dependent lifetimes
    # Ref: Klaassen, Solid-State Electronics 35, 125 (1992)

    "tau_max_n": 1.0e-3,  # [s] - Maximum electron lifetime (low doping limit)
    # High-quality silicon value

    "N_ref_n": 1.0e17,  # [cm⁻³] - Reference doping for electron lifetime
    # Ref: Klaassen (1992)

    "tau_max_p": 1.0e-3,  # [s] - Maximum hole lifetime (low doping limit)
    # High-quality silicon value

    "N_ref_p": 1.0e17,  # [cm⁻³] - Reference doping for hole lifetime
    # Ref: Klaassen (1992)

    # ==============================================================================
    # CARRIER MOBILITY PARAMETERS (Caughey-Thomas Model)
    # ==============================================================================
    # Ref: Caughey & Thomas, Proc. IEEE 55, 2192 (1967)
    # Updated values from Masetti et al., IEEE TED-30, 764 (1983)

    "mu_max_n": 1417.0,  # [cm²/V·s] - Maximum electron mobility (low doping)
    # Ref: Jacoboni et al., Solid-State Electronics 20, 77 (1977)

    "mu_min_n": 68.5,  # [cm²/V·s] - Minimum electron mobility (high doping)
    # Ref: Masetti et al. (1983)

    "N_ref_mob_n": 9.20e16,  # [cm⁻³] - Reference doping for electron mobility
    # Ref: Masetti et al. (1983)

    "alpha_mob_n": 0.711,  # [dimensionless] - Exponent for electron mobility model
    # Ref: Masetti et al. (1983)

    "mu_max_p": 470.5,  # [cm²/V·s] - Maximum hole mobility (low doping)
    # Ref: Jacoboni et al. (1977)

    "mu_min_p": 44.9,  # [cm²/V·s] - Minimum hole mobility (high doping)
    # Ref: Masetti et al. (1983)

    "N_ref_mob_p": 2.23e17,  # [cm⁻³] - Reference doping for hole mobility
    # Ref: Masetti et al. (1983)

    "alpha_mob_p": 0.719,  # [dimensionless] - Exponent for hole mobility model
    # Ref: Masetti et al. (1983)

    # ==============================================================================
    # VELOCITY SATURATION PARAMETERS
    # ==============================================================================
    # Ref: Canali et al., IEEE TED-22, 1045 (1975)

    "v_sat_n": 1.07e7,  # [cm/s] - Electron saturation velocity
    # Ref: Canali et al. (1975)

    "beta_n": 2.0,  # [dimensionless] - Field dependence exponent for electrons
    # Ref: Caughey & Thomas (1967)

    "v_sat_p": 8.37e6,  # [cm/s] - Hole saturation velocity
    # Ref: Canali et al. (1975)

    "beta_p": 1.0,  # [dimensionless] - Field dependence exponent for holes
    # Ref: Caughey & Thomas (1967)

    # ==============================================================================
    # TRAP-ASSISTED TUNNELING (TAT) PARAMETERS - Hurkx Model
    # ==============================================================================
    # Ref: Hurkx et al., IEEE TED-39, 331 (1992)

    "N_t_TAT": 1e14,  # [cm⁻³] - Trap density for TAT
    # Typical for high-quality silicon

    "E_t_TAT": 0.0,  # [eV] - Trap energy level relative to midgap
    # 0.0 = midgap trap (worst case)

    "gamma_TAT": 2.0,  # [dimensionless] - Field enhancement factor
    # Ref: Hurkx et al. (1992)

    "delta_TAT": 3.5e-8,  # [cm/V] - Tunneling parameter
    # Ref: Hurkx et al. (1992)

    # ==============================================================================
    # IMPACT IONIZATION PARAMETERS - Chynoweth Model
    # ==============================================================================
    # Ref: Van Overstraeten & De Man, Solid-State Electronics 13, 583 (1970)
    # Updated values from Maes et al., IEEE TED-37, 2098 (1990)

    "a_n": 7.03e5,  # [cm⁻¹] - Electron impact ionization coefficient prefactor
    # Ref: Maes et al. (1990)

    "b_n": 1.231e6,  # [V/cm] - Electron impact ionization critical field
    # Ref: Maes et al. (1990)

    "a_p": 6.71e5,  # [cm⁻¹] - Hole impact ionization coefficient prefactor
    # Ref: Maes et al. (1990)

    "b_p": 1.693e6,  # [V/cm] - Hole impact ionization critical field
    # Ref: Maes et al. (1990)

    "E_th_n": 1.1e6,  # [V/cm] - Electron impact ionization threshold field
    # Approximate threshold for measurable ionization

    "E_th_p": 1.6e6,  # [V/cm] - Hole impact ionization threshold field
    # Approximate threshold for measurable ionization

    # ==============================================================================
    # SURFACE RECOMBINATION VELOCITIES
    # ==============================================================================
    "s_n": 10.0,  # [cm/s] - Electron surface recombination velocity
    # Typical for passivated Si surface (SiO₂)
    # Ref: Aberle, Prog. Photovolt. 8, 473 (2000)

    "s_p": 10.0,  # [cm/s] - Hole surface recombination velocity
    # Typical for passivated Si surface (SiO₂)
    # Ref: Aberle (2000)

    # ==============================================================================
    # AUGER RECOMBINATION COEFFICIENTS
    # ==============================================================================
    # Ref: Dziewior & Schmid, Appl. Phys. Lett. 31, 346 (1977)
    # Updated: Kerr & Cuevas, J. Appl. Phys. 91, 2473 (2002)

    "C_n_auger": 2.8e-31,  # [cm⁶/s] - Auger coefficient for electrons (eeh process)
    # Ref: Kerr & Cuevas (2002)

    "C_p_auger": 9.9e-32,  # [cm⁶/s] - Auger coefficient for holes (ehh process)
    # Ref: Kerr & Cuevas (2002)

    # ==============================================================================
    # BANDGAP NARROWING PARAMETERS - Slotboom Model
    # ==============================================================================
    # Ref: Slotboom & de Graaff, Solid-State Electronics 19, 857 (1976)
    # Updated: Schenk, J. Appl. Phys. 84, 3684 (1998)

    "BGN_V0": 0.009,  # [V] - Bandgap narrowing voltage scale
    # Ref: Slotboom & de Graaff (1976)

    "BGN_N_ref": 1.0e17,  # [cm⁻³] - Reference doping for bandgap narrowing
    # Ref: Slotboom & de Graaff (1976)

    # ==============================================================================
    # OPTICAL PROPERTIES - Wavelength-Dependent
    # ==============================================================================
    # Ref: Green & Keevers, Prog. Photovolt. 3, 189 (1995)
    # Ref: Green, Solar Energy Materials & Solar Cells 92, 1305 (2008)

    "optical_data": {
        # Wavelengths in nanometers
        "wavelengths": np.array([
            300, 350, 400, 450, 500, 550, 600, 650, 700, 750,
            800, 850, 900, 950, 1000, 1050, 1100, 1150, 1200
        ]),

        # Refractive index (real part) at 300K
        # Ref: Green (2008), Aspnes & Studna, Phys. Rev. B 27, 985 (1983)
        "n_values": np.array([
            1.70, 4.42, 5.57, 4.67, 4.29, 4.08, 3.93, 3.84, 3.76, 3.71,
            3.68, 3.66, 3.64, 3.62, 3.61, 3.59, 3.58, 3.57, 3.56
        ]),

        # Extinction coefficient at 300K
        # Ref: Green (2008), Weakliem & Redfield, J. Appl. Phys. 50, 1491 (1979)
        "k_values": np.array([
            3.10, 2.90, 0.38, 0.116, 0.056, 0.030, 0.020, 0.012, 0.008, 0.005,
            0.003, 0.002, 0.0015, 0.001, 0.0008, 0.0005, 0.0003, 0.0001, 0.00005
        ])
    }
}

# ==============================================================================
# GLOBAL SIMULATION PARAMETERS (Device-specific, not material properties)
# ==============================================================================
GLOBAL_PARAMS = {
    "device_name": "photodiode",
    "mesh_file": "output/photodiode_mesh.msh",
    "temperature_K": 300.0,  # [K] - Operating temperature
    "photon_flux": 0.0,  # [photons/cm²/s] - Incident photon flux (0 for dark)
    "wavelength_start_nm": 400,  # [nm] - Start wavelength for spectral sweep
    "wavelength_end_nm": 1100,  # [nm] - End wavelength for spectral sweep
    "wavelength_points": 71  # Number of wavelength points for spectral sweep
}