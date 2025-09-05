#utils.py
import numpy as np


def get_alpha_for_wavelength(wavelength_nm, material_params):
    """
    Calculates alpha from the extinction coefficient k_e.
    """
    n_r, k_e = get_silicon_optical_constants_lookup(wavelength_nm, material_params)
    alpha_cm = (4 * np.pi * k_e) / (wavelength_nm * 1e-7)  # wavelength converted to cm
    return alpha_cm

def get_silicon_optical_constants_lookup(wavelength_nm, material_params):
    """
    Calculates n_r and k_e for the material using a lookup table and interpolation.
    """
    optical_data = material_params["optical_data"]
    n_r = np.interp(wavelength_nm, optical_data["wavelengths"], optical_data["n_values"])
    k_e = np.interp(wavelength_nm, optical_data["wavelengths"], optical_data["k_values"])
    return (n_r, k_e)