# pn_sim/utils.py
"""
Utility functions for optical calculations and data lookup.
"""
import numpy as np

def get_alpha_for_wavelength(wavelength_nm, material_params):
    """Interpolates the absorption coefficient (alpha) from material data."""
    optical_data = material_params["optical_data"]
    alpha = np.interp(wavelength_nm, optical_data["wavelengths_nm"], optical_data["alpha_cm"])
    return alpha

def get_reflectivity(wavelength_nm, material_params):
    """Calculates surface reflectivity R = |(n-1)/(n+1)|^2, ignoring extinction."""
    optical_data = material_params["optical_data"]
    n_r = np.interp(wavelength_nm, optical_data["wavelengths_nm"], optical_data["n_refractive"])
    return ((n_r - 1) / (n_r + 1))**2
