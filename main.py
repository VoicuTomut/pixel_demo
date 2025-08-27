# main.py
from devsim_simulation import DeviceSimulator
import matplotlib.pyplot as plt
import os
import devsim

if __name__ == "__main__":
    print("Starting Final Robust Photodiode Simulation...")
    try:
        # 1. Create the simulator instance. This sets up the device.
        # This will fail if the mesh file doesn't exist. Run create_mesh.py first.
        simulator = DeviceSimulator()

        # 2. Run the I-V sweep.
        voltages, currents = simulator.run_iv_sweep(v_start=0.0, v_stop=-5.0, num_points=26)

        # 3. Plot the results
        plt.figure(figsize=(8, 6))
        plt.plot(voltages, currents, 'o-', label="Dark Current")
        plt.yscale('log')
        plt.xlabel("Voltage (V)")
        plt.ylabel("Current (A)")
        plt.title("Photodiode I-V Characteristic")
        plt.grid(True, which="both", ls="--")
        plt.legend()

        if not os.path.exists("output"):
            os.makedirs("output")
        plot_path = "output/iv_curve.png"
        plt.savefig(plot_path)
        print(f"\nI-V curve plot saved to {plot_path}")

        # 4. Export final state for visualization
        print("Exporting final device state to output/final_state.vtk...")
        devsim.write_devices(file="output/final_state.vtk", type="vtk")

        print("\n" + "=" * 50)
        print("✅ SIMULATION COMPLETED SUCCESSFULLY")
        print("=" * 50)

    except Exception as e:
        print(f"\nANALYSIS FAILED: {e}")
        import traceback

        traceback.print_exc()