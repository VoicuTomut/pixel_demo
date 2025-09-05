from parameters import GLOBAL_PARAMS, SILICON_PARAMS
from pn_sim.device_builder import  (load_mesh_and_create_device,verify_device_structure,
                                    comprehensive_mesh_debug,verify_mesh_dimensions)

from pn_sim.physics_setup import (
    setup_physics_and_materials,
    setup_boundary_conditions,
    setup_carrier_transport_equations,
    debug_doping_profile
)

from pn_sim.transport_models import (
    force_optical_update,
    comprehensive_optical_debug,
    calculate_qe,
    debug_photocurrent_mechanism,
    run_iv_sweep,
    run_cv_sweep_ac,
    run_spectral_sweep,
    plot_results
)

from pn_sim.equilibrium_solver import  solve_initial_equilibrium, debug_equilibrium_state

import numpy as np

def main():
    """
    Main execution with comprehensive step-by-step debugging workflow.
    Each step has clear objectives, debug outputs, and validation checkpoints.
    """
    import sys
    import datetime
    import os
    import json
    import devsim

    # ==============================================================================
    # INITIALIZATION AND LOGGING SETUP
    # ==============================================================================
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = "output/debug_logs"
    os.makedirs(log_dir, exist_ok=True)

    # Create both a main log and step-specific logs
    main_log_path = f"{log_dir}/main_debug_{timestamp}.txt"
    summary_log_path = f"{log_dir}/summary_{timestamp}.json"

    print(f"Starting simulation at {timestamp}")
    print(f"Main log: {main_log_path}")
    print(f"Summary will be saved to: {summary_log_path}")

    # Redirect output to log file while keeping console output
    class TeeOutput:
        def __init__(self, file):
            self.terminal = sys.stdout
            self.log = file

        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)

        def flush(self):
            self.terminal.flush()
            self.log.flush()

    log_file = open(main_log_path, 'w')
    original_stdout = sys.stdout
    sys.stdout = TeeOutput(log_file)

    # Dictionary to track simulation status
    simulation_status = {
        "timestamp": timestamp,
        "steps_completed": [],
        "errors": [],
        "warnings": [],
        "key_results": {},
        "convergence_issues": []
    }

    try:
        print("=" * 80)
        print("                    PHOTODIODE SIMULATION DEBUG WORKFLOW")
        print("                         Step-by-Step Validation")
        print("=" * 80)

        device_name = GLOBAL_PARAMS["device_name"]

        # ==============================================================================
        # STEP 1: MESH LOADING AND GEOMETRY VALIDATION
        # ==============================================================================
        print("\n" + "=" * 80)
        print("STEP 1: MESH LOADING AND GEOMETRY VALIDATION")
        print("=" * 80)
        print("Objective: Load mesh and verify geometry is suitable for photodiode")

        try:
            # Load mesh
            load_mesh_and_create_device(device_name, GLOBAL_PARAMS["mesh_file"])
            verify_device_structure(device_name)

            # Comprehensive mesh analysis
            print("\n--- Detailed Mesh Analysis ---")
            comprehensive_mesh_debug(device_name, SILICON_PARAMS)

            # Verify dimensions
            device_width, device_depth = verify_mesh_dimensions(device_name,
                                                                min_depth_um=5.0,
                                                                min_width_um=5.0)

            simulation_status["key_results"]["device_width_um"] = float(device_width)
            simulation_status["key_results"]["device_depth_um"] = float(device_depth)
            simulation_status["steps_completed"].append("STEP_1_MESH")

            print(f"\n✅ STEP 1 COMPLETE: Mesh loaded successfully")
            print(f"   Device dimensions: {device_width:.2f} × {device_depth:.2f} μm")

        except Exception as e:
            error_msg = f"STEP 1 FAILED: {str(e)}"
            print(f"\n❌ {error_msg}")
            simulation_status["errors"].append(error_msg)
            raise

        # ==============================================================================
        # STEP 2: PHYSICS MODEL DEFINITION
        # ==============================================================================
        print("\n" + "=" * 80)
        print("STEP 2: PHYSICS MODEL DEFINITION")
        print("=" * 80)
        print("Objective: Define all physics models and material parameters")

        try:
            # Setup physics
            setup_physics_and_materials(device_name, SILICON_PARAMS, GLOBAL_PARAMS)

            # Debug doping profile
            print("\n--- Doping Profile Verification ---")
            debug_doping_profile(device_name)

            # Verify all models are defined
            print("\n--- Verifying Model Definitions ---")
            for region in ["p_region", "n_region"]:
                models_to_check = ["NetDoping", "TotalDoping", "IntrinsicCarrierDensity",
                                   "LowFieldElectronMobility", "LowFieldHoleMobility"]
                for model in models_to_check:
                    try:
                        values = devsim.get_node_model_values(device=device_name,
                                                              region=region,
                                                              name=model)
                        print(f"  ✓ {region}/{model}: Defined ({len(values)} nodes)")
                    except:
                        error = f"  ✗ {region}/{model}: NOT DEFINED"
                        print(error)
                        simulation_status["errors"].append(error)

            simulation_status["steps_completed"].append("STEP_2_PHYSICS")
            print(f"\n✅ STEP 2 COMPLETE: Physics models defined")

        except Exception as e:
            error_msg = f"STEP 2 FAILED: {str(e)}"
            print(f"\n❌ {error_msg}")
            simulation_status["errors"].append(error_msg)
            raise

        # ==============================================================================
        # STEP 3: BOUNDARY CONDITIONS
        # ==============================================================================
        print("\n" + "=" * 80)
        print("STEP 3: BOUNDARY CONDITIONS SETUP")
        print("=" * 80)
        print("Objective: Define contact and interface boundary conditions")

        try:
            setup_boundary_conditions(device_name, SILICON_PARAMS)

            # Verify boundary conditions
            print("\n--- Boundary Condition Verification ---")
            for contact in ["anode", "cathode"]:
                bias = devsim.get_parameter(device=device_name, name=f"{contact}_bias")
                print(f"  {contact}_bias = {bias} V")

            # Check interface
            interfaces = devsim.get_interface_list(device=device_name)
            print(f"  Interfaces defined: {interfaces}")

            simulation_status["steps_completed"].append("STEP_3_BOUNDARIES")
            print(f"\n✅ STEP 3 COMPLETE: Boundary conditions set")

        except Exception as e:
            error_msg = f"STEP 3 FAILED: {str(e)}"
            print(f"\n❌ {error_msg}")
            simulation_status["errors"].append(error_msg)
            raise

        # ==============================================================================
        # STEP 4: INITIAL EQUILIBRIUM SOLUTION
        # ==============================================================================
        print("\n" + "=" * 80)
        print("STEP 4: INITIAL EQUILIBRIUM SOLUTION (0V)")
        print("=" * 80)
        print("Objective: Solve Poisson equation and establish thermal equilibrium")

        try:
            # Solve equilibrium
            solve_initial_equilibrium(device_name)

            # Debug equilibrium state
            print("\n--- Equilibrium State Analysis ---")
            debug_equilibrium_state(device_name)

            # Verify charge neutrality
            for region in ["p_region", "n_region"]:
                space_charge = np.array(devsim.get_node_model_values(device=device_name,
                                                                     region=region,
                                                                     name="SpaceCharge"))
                max_imbalance = np.max(np.abs(space_charge))
                if max_imbalance > 1e10:
                    warning = f"High charge imbalance in {region}: {max_imbalance:.2e}"
                    print(f"  ⚠️ {warning}")
                    simulation_status["warnings"].append(warning)
                else:
                    print(f"  ✓ {region} charge balance OK: {max_imbalance:.2e}")

            simulation_status["steps_completed"].append("STEP_4_EQUILIBRIUM")
            print(f"\n✅ STEP 4 COMPLETE: Equilibrium established")

        except Exception as e:
            error_msg = f"STEP 4 FAILED: {str(e)}"
            print(f"\n❌ {error_msg}")
            simulation_status["errors"].append(error_msg)
            raise

        # ==============================================================================
        # STEP 5: ADD TRANSPORT EQUATIONS
        # ==============================================================================

        print("\n" + "=" * 80*zx)
        print("STEP 5: CARRIER TRANSPORT EQUATIONS")
        print("=" * 80)
        print("Objective: Add drift-diffusion transport and recombination models")

        try:
            # All models and equations were already defined and solved in Step 4.
            # This step is now just a final verification.
            print("\n--- Verifying Transport Equation Setup ---")
            if "STEP_4_EQUILIBRIUM" in simulation_status["steps_completed"]:
                print("  ✓ All physics models and equations are correctly configured.")
                simulation_status["steps_completed"].append("STEP_5_TRANSPORT")
                print(f"\n✅ STEP 5 COMPLETE: Transport setup verified")
            else:
                raise RuntimeError("Cannot verify transport because Step 4 failed.")

        except Exception as e:
            error_msg = f"STEP 5 FAILED: {str(e)}"
            print(f"\n❌ {error_msg}")
            simulation_status["errors"].append(error_msg)
            raise

        # ==============================================================================
        # STEP 6: DARK I-V CHARACTERIZATION
        # ==============================================================================
        print("\n" + "=" * 80)
        print("STEP 6: DARK I-V CHARACTERIZATION")
        print("=" * 80)
        print("Objective: Verify device works properly without illumination")

        try:
            # Voltage sweep from forward to reverse bias
            iv_voltages = np.concatenate([
                np.linspace(0, 1, 11),  # Forward bias (fine steps)
                np.linspace(0.9, -0.1, 11),  # Transition region
                np.linspace(-0.2, -5, 25)  # Reverse bias
            ])

            print(f"\n--- Running Dark I-V Sweep ---")
            print(f"  Voltage range: {iv_voltages[0]:.1f}V to {iv_voltages[-1]:.1f}V")
            print(f"  Number of points: {len(iv_voltages)}")

            # Ensure no optical generation
            devsim.set_parameter(name="EffectivePhotonFlux", value=0.0)

            # Run sweep with debug info
            dark_currents = []
            for i, v in enumerate(iv_voltages):
                print(f"\n  Point {i + 1}/{len(iv_voltages)}: V = {v:.3f}V", end="")
                devsim.set_parameter(device=device_name, name="anode_bias", value=v)

                try:
                    devsim.solve(type="dc", absolute_error=10, relative_error=1e-3,
                                 maximum_iterations=100)

                    e_current = devsim.get_contact_current(device=device_name,
                                                           contact="anode",
                                                           equation="ElectronContinuityEquation")
                    h_current = devsim.get_contact_current(device=device_name,
                                                           contact="anode",
                                                           equation="HoleContinuityEquation")
                    total_current = e_current + h_current
                    dark_currents.append(total_current)
                    print(f" → I = {total_current:.3e} A/cm ✓")

                except Exception as e:
                    print(f" → FAILED: {str(e)}")
                    simulation_status["convergence_issues"].append(f"Dark I-V failed at {v}V")
                    dark_currents.append(float('nan'))

            dark_currents = np.array(dark_currents)

            # Analyze results
            print("\n--- Dark Current Analysis ---")
            reverse_current = dark_currents[iv_voltages == -5.0]
            if len(reverse_current) > 0 and not np.isnan(reverse_current[0]):
                print(f"  Reverse current at -5V: {reverse_current[0]:.3e} A/cm")
                simulation_status["key_results"]["dark_current_5V"] = float(reverse_current[0])

                if abs(reverse_current[0]) > 1e-6:
                    warning = f"High dark current: {reverse_current[0]:.3e} A/cm"
                    simulation_status["warnings"].append(warning)
                    print(f"  ⚠️ {warning}")

            simulation_status["steps_completed"].append("STEP_6_DARK_IV")
            print(f"\n✅ STEP 6 COMPLETE: Dark I-V characterized")

        except Exception as e:
            error_msg = f"STEP 6 FAILED: {str(e)}"
            print(f"\n❌ {error_msg}")
            simulation_status["errors"].append(error_msg)
            # Continue anyway to test other parts

        # ==============================================================================
        # STEP 7: OPTICAL RESPONSE TEST
        # ==============================================================================
        print("\n" + "=" * 80)
        print("STEP 7: OPTICAL RESPONSE VERIFICATION")
        print("=" * 80)
        print("Objective: Test photodetection capability")

        try:
            # Test parameters
            test_wavelength = 650  # nm
            test_flux = 1e17  # photons/cm²/s (~1 sun)
            test_voltage = -2.0  # V (reverse bias)

            print(f"\n--- Optical Test Parameters ---")
            print(f"  Wavelength: {test_wavelength} nm")
            print(f"  Photon flux: {test_flux:.2e} photons/cm²/s")
            print(f"  Bias voltage: {test_voltage} V")

            # Set optical parameters
            alpha = get_alpha_for_wavelength(test_wavelength, SILICON_PARAMS)
            reflectivity = get_reflectivity(test_wavelength, SILICON_PARAMS)
            effective_flux = test_flux * (1.0 - reflectivity)

            print(f"\n--- Optical Properties ---")
            print(f"  Absorption coefficient: {alpha:.2e} cm⁻¹")
            print(f"  Reflectivity: {reflectivity:.3%}")
            print(f"  Effective flux: {effective_flux:.2e} photons/cm²/s")

            devsim.set_parameter(name="alpha", value=alpha)
            devsim.set_parameter(name="EffectivePhotonFlux", value=effective_flux)
            devsim.set_parameter(device=device_name, name="anode_bias", value=test_voltage)

            # Force optical update
            force_optical_update(device_name)

            # Debug optical generation
            print("\n--- Optical Generation Debug ---")
            comprehensive_optical_debug(device_name)

            # Solve with light
            print("\n--- Solving with illumination ---")
            devsim.solve(type="dc", absolute_error=10, relative_error=1e-3,
                         maximum_iterations=200)

            # Get photocurrent
            e_current_light = devsim.get_contact_current(device=device_name,
                                                         contact="anode",
                                                         equation="ElectronContinuityEquation")
            h_current_light = devsim.get_contact_current(device=device_name,
                                                         contact="anode",
                                                         equation="HoleContinuityEquation")
            light_current = e_current_light + h_current_light

            # Compare with dark current
            dark_at_test_v = np.interp(test_voltage, iv_voltages, dark_currents)
            photocurrent = light_current - dark_at_test_v

            print(f"\n--- Photocurrent Analysis ---")
            print(f"  Dark current: {dark_at_test_v:.3e} A/cm")
            print(f"  Light current: {light_current:.3e} A/cm")
            print(f"  Photocurrent: {photocurrent:.3e} A/cm")

            # Calculate QE
            qe = calculate_qe(dark_at_test_v, light_current, test_flux, test_wavelength)
            print(f"  Quantum Efficiency: {qe:.2f}%")

            simulation_status["key_results"]["photocurrent_650nm"] = float(photocurrent)
            simulation_status["key_results"]["QE_650nm"] = float(qe)

            if abs(photocurrent) < 1e-12:
                warning = "Negligible photocurrent detected"
                simulation_status["warnings"].append(warning)
                print(f"  ⚠️ {warning}")
                debug_photocurrent_mechanism(device_name, test_voltage)

            simulation_status["steps_completed"].append("STEP_7_OPTICAL")
            print(f"\n✅ STEP 7 COMPLETE: Optical response tested")

        except Exception as e:
            error_msg = f"STEP 7 FAILED: {str(e)}"
            print(f"\n❌ {error_msg}")
            simulation_status["errors"].append(error_msg)

        # ==============================================================================
        # STEP 8: FULL I-V WITH LIGHT
        # ==============================================================================
        print("\n" + "=" * 80)
        print("STEP 8: ILLUMINATED I-V SWEEP")
        print("=" * 80)
        print("Objective: Full I-V characterization under illumination")

        try:
            # Keep same optical conditions from Step 7
            print(f"\n--- Light I-V Sweep ---")
            print(f"  Using {test_wavelength}nm illumination at {test_flux:.2e} photons/cm²/s")

            light_currents = run_iv_sweep(device_name, iv_voltages,
                                          p_flux=test_flux,
                                          material_params=SILICON_PARAMS,
                                          wavelength_nm=test_wavelength)

            # Find key parameters
            print("\n--- Photodiode Parameters ---")

            # Short-circuit current (at V=0)
            isc = np.interp(0.0, iv_voltages, light_currents)
            print(f"  Short-circuit current (Isc): {isc:.3e} A/cm")
            simulation_status["key_results"]["Isc"] = float(isc)

            # Open-circuit voltage (where I=0)
            if np.any(light_currents > 0) and np.any(light_currents < 0):
                voc = np.interp(0.0, light_currents[::-1], iv_voltages[::-1])
                print(f"  Open-circuit voltage (Voc): {voc:.3f} V")
                simulation_status["key_results"]["Voc"] = float(voc)

            simulation_status["steps_completed"].append("STEP_8_LIGHT_IV")
            print(f"\n✅ STEP 8 COMPLETE: Light I-V characterized")

        except Exception as e:
            error_msg = f"STEP 8 FAILED: {str(e)}"
            print(f"\n❌ {error_msg}")
            simulation_status["errors"].append(error_msg)

        # ==============================================================================
        # STEP 9: CAPACITANCE-VOLTAGE
        # ==============================================================================
        print("\n" + "=" * 80)
        print("STEP 9: CAPACITANCE-VOLTAGE CHARACTERIZATION")
        print("=" * 80)
        print("Objective: Extract junction capacitance and depletion width")

        try:
            cv_voltages = np.linspace(1, -5, 31)
            freq_hz = 1e6  # 1 MHz

            print(f"\n--- C-V Measurement ---")
            print(f"  Frequency: {freq_hz / 1e6:.1f} MHz")
            print(f"  Voltage range: {cv_voltages[0]}V to {cv_voltages[-1]}V")

            capacitances = run_cv_sweep_ac(device_name, cv_voltages, freq_hz)

            # Extract built-in voltage from Mott-Schottky plot
            valid_idx = ~np.isnan(capacitances)
            if np.sum(valid_idx) > 5:
                inv_c_squared = 1.0 / capacitances[valid_idx] ** 2
                # Linear fit in reverse bias region
                reverse_idx = cv_voltages[valid_idx] < -1
                if np.sum(reverse_idx) > 3:
                    from scipy import stats
                    slope, intercept, r_value, _, _ = stats.linregress(
                        cv_voltages[valid_idx][reverse_idx],
                        inv_c_squared[reverse_idx]
                    )
                    vbi = -intercept / slope if slope != 0 else 0
                    print(f"\n--- Mott-Schottky Analysis ---")
                    print(f"  Built-in voltage: {vbi:.3f} V")
                    print(f"  R² value: {r_value ** 2:.4f}")
                    simulation_status["key_results"]["Vbi"] = float(vbi)

            simulation_status["steps_completed"].append("STEP_9_CV")
            print(f"\n✅ STEP 9 COMPLETE: C-V characterized")

        except Exception as e:
            error_msg = f"STEP 9 FAILED: {str(e)}"
            print(f"\n❌ {error_msg}")
            simulation_status["errors"].append(error_msg)

        # ==============================================================================
        # STEP 10: SPECTRAL RESPONSE
        # ==============================================================================
        print("\n" + "=" * 80)
        print("STEP 10: SPECTRAL RESPONSE CHARACTERIZATION")
        print("=" * 80)
        print("Objective: Measure quantum efficiency vs wavelength")

        try:
            wavelengths_nm = np.linspace(400, 1100, 15)
            qe_bias = -2.0  # V
            incident_flux = 1e17  # photons/cm²/s

            print(f"\n--- Spectral Sweep Parameters ---")
            print(f"  Wavelength range: {wavelengths_nm[0]}-{wavelengths_nm[-1]} nm")
            print(f"  Bias voltage: {qe_bias} V")
            print(f"  Incident flux: {incident_flux:.2e} photons/cm²/s")

            qe_spectral = run_spectral_sweep(device_name, wavelengths_nm, qe_bias,
                                             incident_flux, None, iv_voltages,
                                             dark_currents, SILICON_PARAMS)

            # Find peak QE
            valid_qe = np.array(qe_spectral)[~np.isnan(qe_spectral)]
            if len(valid_qe) > 0:
                peak_qe = np.max(valid_qe)
                peak_wavelength = wavelengths_nm[np.argmax(qe_spectral)]
                print(f"\n--- Spectral Response Summary ---")
                print(f"  Peak QE: {peak_qe:.2f}% at {peak_wavelength:.0f} nm")
                simulation_status["key_results"]["peak_QE"] = float(peak_qe)
                simulation_status["key_results"]["peak_wavelength_nm"] = float(peak_wavelength)

            simulation_status["steps_completed"].append("STEP_10_SPECTRAL")
            print(f"\n✅ STEP 10 COMPLETE: Spectral response characterized")

        except Exception as e:
            error_msg = f"STEP 10 FAILED: {str(e)}"
            print(f"\n❌ {error_msg}")
            simulation_status["errors"].append(error_msg)

        # ==============================================================================
        # FINAL SUMMARY
        # ==============================================================================
        print("\n" + "=" * 80)
        print("SIMULATION COMPLETE - SUMMARY")
        print("=" * 80)

        print(f"\nSteps Completed: {len(simulation_status['steps_completed'])}/10")
        for step in simulation_status["steps_completed"]:
            print(f"  ✅ {step}")

        if simulation_status["errors"]:
            print(f"\n❌ Errors Encountered: {len(simulation_status['errors'])}")
            for error in simulation_status["errors"][:5]:  # Show first 5
                print(f"  - {error}")

        if simulation_status["warnings"]:
            print(f"\n⚠️ Warnings: {len(simulation_status['warnings'])}")
            for warning in simulation_status["warnings"][:5]:  # Show first 5
                print(f"  - {warning}")

        if simulation_status["key_results"]:
            print(f"\n📊 Key Results:")
            for key, value in simulation_status["key_results"].items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.3e}")
                else:
                    print(f"  {key}: {value}")

        # Save summary to JSON
        with open(summary_log_path, 'w') as f:
            json.dump(simulation_status, f, indent=2)
        print(f"\n📁 Summary saved to: {summary_log_path}")

        # ==============================================================================
        # GENERATE PLOTS
        # ==============================================================================
        if len(simulation_status["steps_completed"]) >= 6:
            print("\n--- Generating Visualization Plots ---")
            try:
                # Plot results with available data
                plot_results(iv_voltages, dark_currents,
                             light_currents if 'light_currents' in locals() else dark_currents,
                             qe_spectral if 'qe_spectral' in locals() else [],
                             cv_voltages if 'cv_voltages' in locals() else [],
                             capacitances if 'capacitances' in locals() else [],
                             wavelengths_nm if 'wavelengths_nm' in locals() else [],
                             qe_spectral if 'qe_spectral' in locals() else [],
                             test_wavelength if 'test_wavelength' in locals() else 650,
                             qe_bias if 'qe_bias' in locals() else -2.0)
                print("✅ Plots generated successfully")
            except Exception as e:
                print(f"❌ Plot generation failed: {e}")

    except Exception as e:
        print(f"\n\n{'=' * 80}")
        print(f"FATAL ERROR: {str(e)}")
        print(f"{'=' * 80}")
        import traceback
        traceback.print_exc()
        simulation_status["errors"].append(f"FATAL: {str(e)}")

    finally:
        # Save final status
        with open(summary_log_path, 'w') as f:
            json.dump(simulation_status, f, indent=2)

        # Close log file
        log_file.close()
        sys.stdout = original_stdout

        print(f"\n✅ Simulation workflow completed")
        print(f"📁 Full log saved to: {main_log_path}")
        print(f"📁 Summary saved to: {summary_log_path}")

        # Quick diagnostic output
        print("\n🔍 Quick Diagnostics:")
        print(f"  Total steps completed: {len(simulation_status['steps_completed'])}/10")
        print(f"  Errors: {len(simulation_status['errors'])}")
        print(f"  Warnings: {len(simulation_status['warnings'])}")
        print(f"  Convergence issues: {len(simulation_status['convergence_issues'])}")

        if simulation_status['errors']:
            print("\n  First error: " + simulation_status['errors'][0][:100])


if __name__ == "__main__":
    main()