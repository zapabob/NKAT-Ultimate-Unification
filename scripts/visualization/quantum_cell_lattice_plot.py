import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import sys

# --- Utility: find latest result file ---
def find_latest_result(pattern="quantum_cell_sim_results_*.npz"):
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError("No simulation result file found.")
    return max(files, key=os.path.getmtime)

# --- Load data ---
def load_data(filename=None):
    if filename is None:
        filename = find_latest_result()
    data = np.load(filename)
    energies = data["energies"]
    entropies = data["entropies"]
    entanglements = data["entanglements"] if "entanglements" in data else None
    return energies, entropies, entanglements, filename

# --- Plot time series ---
def plot_time_series(energies, entropies, entanglements, out_prefix):
    plt.figure(figsize=(10,6))
    plt.plot(energies, label="Energy")
    plt.plot(entropies, label="Entropy")
    if entanglements is not None:
        plt.plot(entanglements, label="Entanglement")
    plt.xlabel("Time step")
    plt.ylabel("Value")
    plt.title("Quantum Cell Lattice Simulation: Time Series")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_timeseries.png")
    plt.show()

# --- Plot 3D lattice slice heatmaps ---
def plot_lattice_slices(npzfile, out_prefix):
    # Try to load the final lattice state from backup (if available)
    import pickle
    import re
    # Find latest backup for this run
    base = re.sub(r"quantum_cell_sim_results_.*", "", npzfile)
    backup_dir = "backup_archive/quantum_cell_sim/"
    backups = [f for f in os.listdir(backup_dir) if f.endswith(".pkl")]
    if not backups:
        print("[WARN] No backup found for lattice visualization.")
        return
    latest = max(backups, key=lambda x: os.path.getmtime(os.path.join(backup_dir, x)))
    with open(os.path.join(backup_dir, latest), "rb") as f:
        d = pickle.load(f)
    lattice = d["state"]
    N = lattice.shape[0]
    # Plot |00> component (index 0) for z=0, z=N//2, z=N-1
    for z in [0, N//2, N-1]:
        slice_data = np.abs(lattice[:,:,z,0])**2 if lattice.ndim==4 else np.abs(lattice[:,:,z][...,0])**2
        plt.figure(figsize=(6,5))
        plt.imshow(slice_data, cmap="viridis", origin="lower")
        plt.colorbar(label="|<00|Psi>|^2")
        plt.title(f"|00> Component at z={z}")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.tight_layout()
        plt.savefig(f"{out_prefix}_slice_z{z}.png")
        plt.show()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Visualize Quantum Cell Lattice Simulation Results")
    parser.add_argument("--file", type=str, default=None, help="Result npz file to visualize")
    args = parser.parse_args()
    energies, entropies, entanglements, fname = load_data(args.file)
    out_prefix = os.path.splitext(os.path.basename(fname))[0]
    plot_time_series(energies, entropies, entanglements, out_prefix)
    plot_lattice_slices(fname, out_prefix) 