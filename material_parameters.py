# Dictionary of silicon properties.
# Parameters marked with a star (⭐) are those whose fundamental physics
# can be computed from first principles (ab initio) using quantum mechanical simulations.
# These are the key levers for materials discovery and design.

silicon_material_properties = {
    # --- Intrinsic and General Properties ---
    "Permittivity": {#⭐
        "value": 11.9 * 8.854e-14,
        "unit": "F/cm",
        "computable_ab_initio": True,
        "explanation": "Computable as the material's dielectric tensor. A different material with a higher permittivity could be used to engineer junction capacitance, making it a target for materials design."
    },
    "IntrinsicCarrierDensity": {
        "value": 1.0e10,
        "unit": "cm⁻³",
        "computable_ab_initio": True,
        "explanation": "Derived from the **band gap (Eg)** and **carrier effective masses (m*)**, which are fundamental outputs of Quantum Espresso. Searching for a material with a lower band gap is the primary way to increase dark current or absorb longer wavelength light."
    },
    "ElectronCharge": {
        "value": 1.6e-19,
        "unit": "C",
        "computable_ab_initio": False,
        "explanation": "A fundamental physical constant, not a material property. It is the same for all materials."
    },

    # --- Recombination Properties ---
    "Tau": {
        "value": 1.0e-6,
        "unit": "s",
        "computable_ab_initio": True,
        "explanation": "While the macroscopic value isn't computed directly, the underlying physics are. Simulations can determine the effectiveness of specific defects as recombination centers by calculating their **carrier capture cross-sections (σ)**. A 'better' material would be one designed to have defects with smaller capture cross-sections, leading to longer lifetimes."
    },

    # --- Optical Properties ---
    "alpha": {# (Absorption Coefficient) ⭐
        "value": 1e4,
        "unit": "cm⁻¹",
        "computable_ab_initio": True,
        "explanation": "Derived from the **complex dielectric function**, which is calculated by Quantum Espresso. A better photodiode material would have a higher alpha at the target wavelength, allowing it to be thinner and more efficient."
    },

    # --- Mobility Model Parameters (Caughey-Thomas Model) ⭐ ---
    "Electron & Hole Mobility ⭐": {
        "model": "Caughey-Thomas",
        "computable_ab_initio": True,
        "explanation": "The empirical fitting parameters themselves aren't the direct output. Instead, LS-QUANT can directly calculate **mobility vs. doping concentration (μ vs. N)** by simulating carrier scattering from phonons and ionized dopants. You can then fit these first-principles results to the model. A better material would exhibit higher mobility at high doping concentrations."
    }

}
