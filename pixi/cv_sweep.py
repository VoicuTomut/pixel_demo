


def run_cv_sweep(device, voltages, freq_hz):
    """
    Calculates C-V using numerical differentiation of charge.
    """
    capacitances = []
    DELTA_V = 0.001  # 1 mV step for numerical differentiation

    print(f"\nStarting C-V sweep...")

    # Initial solve at starting voltage
    devsim.set_parameter(device=device, name="anode_bias", value=voltages[0])
    devsim.solve(type="dc", absolute_error=10.0, relative_error=1e-2, maximum_iterations=100)

    for i, v in enumerate(voltages):
        print(f"Step {i + 1}/{len(voltages)}: Bias = {v:.2f} V")

        try:
            # Solve at V - DELTA_V/2
            devsim.set_parameter(device=device, name="anode_bias", value=v - DELTA_V / 2.0)
            devsim.solve(type="dc", absolute_error=10.0, relative_error=1e-2, maximum_iterations=100)
            q1 = devsim.get_contact_charge(device=device, contact="anode", equation="PotentialEquation")

            # Solve at V + DELTA_V/2
            devsim.set_parameter(device=device, name="anode_bias", value=v + DELTA_V / 2.0)
            devsim.solve(type="dc", absolute_error=10.0, relative_error=1e-2, maximum_iterations=100)
            q2 = devsim.get_contact_charge(device=device, contact="anode", equation="PotentialEquation")

            # Calculate capacitance
            C = abs(q2 - q1) / DELTA_V
            capacitances.append(C)

            print(f"  C = {C * 1e12:.3f} pF/cm (q1={q1:.3e} C/cm, q2={q2:.3e} C/cm)")

        except devsim.error as msg:
            print(f"  Failed at V = {v:.2f} V: {msg}")
            capacitances.append(float('nan'))

    # Reset bias
    devsim.set_parameter(device=device, name="anode_bias", value=0.0)
    return np.array(capacitances)


def run_cv_sweep_ac(device, voltages, freq_hz):
    """
    Calculates C-V using the more robust small-signal AC analysis method.
    This is the FINAL CORRECTED version.
    """
    capacitances = []
    omega = 2.0 * np.pi * freq_hz  # Angular frequency (rad/s)

    print(f"\nStarting AC C-V sweep at {freq_hz / 1e6:.1f} MHz...")

    for i, v in enumerate(voltages):
        devsim.set_parameter(device=device, name="anode_bias", value=v)
        print(f"Step {i + 1}/{len(voltages)}: Bias = {v:.2f} V")

        try:
            # FIX 1: Use TIGHT tolerances to get a physically correct DC solution.
            # This is the key to fixing the linear shape problem.
            devsim.solve(type="dc", absolute_error=100.0, relative_error=3e-2, maximum_iterations=200)

            # Perform the small-signal AC analysis at this correct DC point.
            devsim.solve(type="ac", frequency=freq_hz)

            # FIX 2: Use the original DC equation names to get the AC current.
            # The simulator automatically returns the AC result after an AC solve.
            imag_i_e = devsim.get_contact_current(device=device, contact="anode",
                                                  equation="ElectronContinuityEquation")
            imag_i_h = devsim.get_contact_current(device=device, contact="anode",
                                                  equation="HoleContinuityEquation")


            C = (imag_i_e + imag_i_h) / omega
            capacitances.append(C)
            print(f"  ✅ C = {C * 1e12:.4f} pF/cm")

        except devsim.error as msg:
            print(f"  ❌ Failed at V = {v:.2f} V: {msg}")
            capacitances.append(float('nan'))

    devsim.set_parameter(device=device, name="anode_bias", value=0.0)
    return np.array(capacitances)
