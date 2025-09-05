# pn_sim/parameters.py
"""
Central repository for all material and global simulation parameters.
This allows for easy modification and ensures consistency across the simulation.
References: S. M. Sze & Kwok K. Ng, "Physics of Semiconductor Devices", 3rd Ed., Wiley, 2007.
"""
import numpy as np

# All values are for Silicon at 300K unless otherwise specified.
SILICON_PARAMS = {
    # Fundamental constants
    "permittivity": 11.7 * 8.854e-14,  # [F/cm] - Silicon relative permittivity * vacuum permittivity
    "electron_charge": 1.60217662e-19,  # [C] - Elementary charge

    # Intrinsic properties at 300K
    "bandgap": 1.12,  # [eV] - Energy gap (Sze, Appendix G)
    "n_i": 1.0e10,  # [cm^-3] - Intrinsic carrier concentration (DEVSIM default for Si)
    "Nc_300K": 2.8e19,  # [cm^-3] - Conduction band effective density of states (Sze, Appendix G)
    "Nv_300K": 1.04e19,  # [cm^-3] - Valence band effective density of states (Sze, Appendix G)

    # Doping profile for the photodiode
    "peak_p_doping": 1e18,  # [cm^-3] - Peak acceptor concentration for the Gaussian profile
    "n_bulk_doping": 1e15,  # [cm^-3] - Uniform background donor concentration

    # Carrier lifetime (SRH recombination)
    "tau_n": 1e-7,  # [s] - Electron lifetime
    "tau_p": 1e-7,  # [s] - Hole lifetime

    # Low-field mobility (cm^2/V-s converted to m^2/V-s in physics_setup)
    "mu_n_low_field": 1350.0,  # [cm^2/V-s] - Electron mobility
    "mu_p_low_field": 450.0,  # [cm^2/V-s] - Hole mobility

    # Optical properties (wavelength-dependent, interpolated)
    # Ref: Green and Keevers, Prog. Photovolt. 3, 189 (1995) for alpha
    "optical_data": {
        "wavelengths_nm": np.array([400, 500, 600, 650, 700, 800, 900, 1000, 1100]),
        "alpha_cm": np.array([1.9e5, 1.2e4, 5.5e3, 3.8e3, 2.7e3, 1.1e3, 3.5e2, 8.5e1, 1e1]),
        "n_refractive": np.array([5.57, 4.29, 3.93, 3.84, 3.76, 3.68, 3.64, 3.61, 3.58])
        # Real part of refractive index
    }
}

GLOBAL_PARAMS = {
    "device_name": "photodiode",
    "mesh_file": "output/photodiode_mesh.msh",
    "temperature_K": 300.0
}