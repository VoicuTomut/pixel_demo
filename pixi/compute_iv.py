
import devsim
import numpy as np

def run_iv_sweep(device, voltages, p_flux):
    currents = []
    devsim.set_parameter(name="PhotonFlux", value=p_flux)
    if p_flux>0:
        print(f"DEBUG: PhotonFlux set to {devsim.get_parameter(name='PhotonFlux')}")  # Add this line

    for v in voltages:
        print(f"\nSetting Anode Bias: {v:.3f} V")
        devsim.set_parameter(device=device, name="anode_bias", value=v)
        try:
            # Add maximum_divergence and increase maximum_iterations
            devsim.solve(type="dc", absolute_error=1e10, relative_error=10,
                         maximum_iterations=400,  # Increased from 300
                         maximum_divergence=10)  # New parameter
            e_current = devsim.get_contact_current(device=device, contact="anode",
                                                   equation="ElectronContinuityEquation")
            h_current = devsim.get_contact_current(device=device, contact="anode", equation="HoleContinuityEquation")
            currents.append(e_current + h_current)
            print(f"✅ V = {v:.3f} V, Current = {currents[-1]:.4e} A/cm")
        except devsim.error as msg:
            # Catch the error, report it, and append a NaN
            print(f"❌ CONVERGENCE FAILED at V = {v:.3f} V. Error: {msg}")
            currents.append(float('nan'))
            # Optional: break the loop if one failure is enough
            break

    devsim.set_parameter(device=device, name="anode_bias", value=0.0)  # Reset bias at the end
    return np.array(currents)

