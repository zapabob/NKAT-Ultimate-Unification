import numpy as np
import cupy as cp
from tqdm import tqdm
import os
import signal
import pickle
import time

# --- Config ---
USE_CUPY = False  # True: use CuPy (GPU), False: use NumPy (CPU)
backend = cp if USE_CUPY else np

# Simulation parameters
N = 8  # 3D lattice size (N x N x N)
T = 100  # Number of time steps
dt = 0.05  # Time step size
J = 1.0  # Interaction strength
THETA = 0.2  # Noncommutative correction strength
BACKUP_INTERVAL = 60  # seconds
BACKUP_DIR = "./backup_archive/quantum_cell_sim/"
os.makedirs(BACKUP_DIR, exist_ok=True)

cell_dim = 4  # 2-bit quantum cell basis: |00>, |01>, |10>, |11>

# Pauli matrices for interaction (tensor product for 2 qubits)
sigma_x = backend.array([[0, 1], [1, 0]], dtype=complex)
sigma_y = backend.array([[0, -1j], [1j, 0]], dtype=complex)
sigma_z = backend.array([[1, 0], [0, -1]], dtype=complex)
I2 = backend.eye(2, dtype=complex)
paulis = [sigma_x, sigma_y, sigma_z]

# Operators on 2-qubit (4x4)
pauli_ops = [backend.kron(p, I2) + backend.kron(I2, p) for p in paulis]

# Moyal-type noncommutative correction operator (example: [X,Y])
def moyal_correction():
    # [sigma_x, sigma_y] = 2i sigma_z (on each qubit), extend to 2-qubit
    return THETA * (backend.kron(sigma_x, sigma_y) - backend.kron(sigma_y, sigma_x))

# Initial state: random normalized 4D complex vector for each cell
def random_cell_state():
    v = backend.random.randn(cell_dim) + 1j * backend.random.randn(cell_dim)
    v /= backend.linalg.norm(v)
    return v

def save_backup(state, t, fname=None):
    if USE_CUPY:
        state = backend.asnumpy(state)
    if fname is None:
        fname = f"{BACKUP_DIR}backup_t{t}_{int(time.time())}.pkl"
    with open(fname, "wb") as f:
        pickle.dump({"state": state, "t": t}, f)

def load_latest_backup():
    files = [f for f in os.listdir(BACKUP_DIR) if f.endswith(".pkl")]
    if not files:
        return None, 0
    latest = max(files, key=lambda x: os.path.getmtime(os.path.join(BACKUP_DIR, x)))
    with open(os.path.join(BACKUP_DIR, latest), "rb") as f:
        d = pickle.load(f)
    state = d["state"]
    if USE_CUPY:
        state = backend.array(state)
    return state, d["t"]

# Hamiltonian for a single cell (can be extended)
def cell_hamiltonian():
    # Example: local field + noncommutative correction
    H = backend.zeros((cell_dim, cell_dim), dtype=complex)
    # Add local field if needed
    H += moyal_correction()
    return H

# Interaction Hamiltonian between two cells
def interaction_hamiltonian():
    # Ising-type (sigma_z ⊗ sigma_z) + noncommutative correction
    return J * backend.kron(sigma_z, sigma_z) + moyal_correction()

# Build full lattice state (N x N x N x 4)
def initialize_lattice(N):
    return backend.stack([
        [ [random_cell_state() for _ in range(N)] for _ in range(N)] for _ in range(N)
    ])

# Time evolution for one step (Trotterized, nearest neighbor)
def time_evolve(lattice, dt):
    N = lattice.shape[0]
    new_lattice = lattice.copy()
    for x in range(N):
        for y in range(N):
            for z in range(N):
                psi = lattice[x, y, z]
                H = cell_hamiltonian()
                # Nearest neighbor interaction
                for dx, dy, dz in [(-1,0,0),(1,0,0),(0,-1,0),(0,1,0),(0,0,-1),(0,0,1)]:
                    nx, ny, nz = x+dx, y+dy, z+dz
                    if 0 <= nx < N and 0 <= ny < N and 0 <= nz < N:
                        neighbor = lattice[nx, ny, nz]
                        # Mean-field + noncommutative correction
                        H += J * backend.outer(psi, neighbor.conj()) * backend.vdot(psi, neighbor)
                        H += moyal_correction()
                # Time evolution (Euler, for demonstration)
                new_lattice[x, y, z] = psi - 1j * dt * H @ psi
                # Renormalize
                new_lattice[x, y, z] /= backend.linalg.norm(new_lattice[x, y, z])
    return new_lattice

# Physical quantities
def compute_total_energy(lattice):
    N = lattice.shape[0]
    E = 0.0
    for x in range(N):
        for y in range(N):
            for z in range(N):
                psi = lattice[x, y, z]
                H = cell_hamiltonian()
                E += backend.real(backend.vdot(psi, H @ psi))
                # Neighbor interaction energy
                for dx, dy, dz in [(-1,0,0),(1,0,0),(0,-1,0),(0,1,0),(0,0,-1),(0,0,1)]:
                    nx, ny, nz = x+dx, y+dy, z+dz
                    if 0 <= nx < N and 0 <= ny < N and 0 <= nz < N:
                        neighbor = lattice[nx, ny, nz]
                        E += J * backend.abs(backend.vdot(psi, neighbor))**2
    return float(E)

def compute_entropy(lattice):
    # Von Neumann entropy for each cell (pure state = 0)
    N = lattice.shape[0]
    S = 0.0
    for x in range(N):
        for y in range(N):
            for z in range(N):
                psi = lattice[x, y, z]
                rho = backend.outer(psi, psi.conj())
                eigs = backend.linalg.eigvalsh(rho)
                S -= backend.sum(eigs * backend.log(eigs + 1e-12))
    return float(S)

# Entanglement entropy (between two neighboring cells)
def compute_pair_entanglement(lattice):
    N = lattice.shape[0]
    ent_list = []
    for x in range(N-1):
        for y in range(N-1):
            for z in range(N-1):
                psi1 = lattice[x, y, z]
                psi2 = lattice[x+1, y, z]
                rho = backend.outer(psi1, psi1.conj()) + backend.outer(psi2, psi2.conj())
                eigs = backend.linalg.eigvalsh(rho)
                S = -backend.sum(eigs * backend.log(eigs + 1e-12))
                ent_list.append(float(S))
    return np.mean(ent_list) if ent_list else 0.0

# Signal handler for emergency save
def signal_handler(sig, frame):
    print("\n[INFO] Emergency backup triggered.")
    save_backup(lattice, t, fname=f"{BACKUP_DIR}emergency_backup_t{t}_{int(time.time())}.pkl")
    print("[INFO] Backup saved. Exiting.")
    exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# Main simulation loop
if __name__ == "__main__":
    # Try to load latest backup
    lattice, t0 = load_latest_backup()
    if lattice is None:
        lattice = initialize_lattice(N)
        t0 = 0
    print(f"[INFO] Simulation start from t={t0}")
    last_backup = time.time()
    energies = []
    entropies = []
    entanglements = []
    for t in tqdm(range(t0, T)):
        lattice = time_evolve(lattice, dt)
        E = compute_total_energy(lattice)
        S = compute_entropy(lattice)
        ent = compute_pair_entanglement(lattice)
        energies.append(E)
        entropies.append(S)
        entanglements.append(ent)
        # Periodic backup
        if time.time() - last_backup > BACKUP_INTERVAL:
            save_backup(lattice, t)
            last_backup = time.time()
        # Optionally: print or log
        if t % 10 == 0:
            print(f"Step {t}: Energy={E:.4f}, Entropy={S:.4f}, Entanglement={ent:.4f}")
    # Final save
    save_backup(lattice, T)
    np.savez(f"quantum_cell_sim_results_{int(time.time())}.npz", energies=energies, entropies=entropies, entanglements=entanglements)
    print("[INFO] Simulation completed and results saved.") 