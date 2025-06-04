#!/usr/bin/env python3
"""
NKAT Consciousness Eigenvalue Solver - Fixed Version
====================================================

Implementation of consciousness operator eigenvalue equations
and GNS construction for universe ground state identification.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from scipy.optimize import minimize
import torch
from tqdm import tqdm
import time
from Chrono import QuantumSpacetimeCell

# Check CUDA availability
CUDA_AVAILABLE = torch.cuda.is_available()
print(f"CUDA Available: {CUDA_AVAILABLE}")
if CUDA_AVAILABLE:
    print(f"GPU: {torch.cuda.get_device_name(0)}")

class ConsciousnessOperator:
    """Consciousness field operator Ψ̂_consciousness"""
    
    def __init__(self, N_modes, N_cutoff):
        """Initialize consciousness operator with finite mode truncation"""
        self.N = N_modes
        self.N_cut = N_cutoff
        self.device = torch.device("cuda" if CUDA_AVAILABLE else "cpu")
        
        # Generate basis functions
        self.basis_functions = self._generate_basis()
        
    def _generate_basis(self):
        """Generate orthonormal basis {Ξ_{mn}} for A_★"""
        basis = []
        
        for m in range(1, self.N + 1):
            for n in range(self.N_cut + 1):
                basis_element = {
                    'mode': m,
                    'level': n,
                    'alpha': 1.0 / (m + n + 1)
                }
                basis.append(basis_element)
        
        return basis
    
    def construct_matrix(self):
        """Construct finite-dimensional matrix representation"""
        size = len(self.basis_functions)
        matrix = torch.zeros((size, size), dtype=torch.float64, device=self.device)
        
        for i, basis_i in enumerate(self.basis_functions):
            for j, basis_j in enumerate(self.basis_functions):
                matrix[i, j] = self._matrix_element(basis_i, basis_j)
        
        return matrix
    
    def _matrix_element(self, basis_i, basis_j):
        """Calculate matrix element between basis functions"""
        m_i, n_i = basis_i['mode'], basis_i['level']
        m_j, n_j = basis_j['mode'], basis_j['level']
        
        # Diagonal terms (energy levels)
        if m_i == m_j and n_i == n_j:
            return (n_i + 0.5) + 0.1 * m_i
        
        # Off-diagonal coupling
        elif abs(m_i - m_j) <= 1 and abs(n_i - n_j) <= 1:
            coupling = 0.01 * np.sqrt(max(n_i, n_j, 1))
            return coupling
        
        return 0.0

class UniverseGroundState:
    """Universe ground state solver using variational methods"""
    
    def __init__(self, consciousness_op):
        """Initialize with consciousness operator"""
        self.psi_con = consciousness_op
        self.device = consciousness_op.device
        
    def variational_ansatz(self, alpha_params):
        """Variational trial state"""
        state_dim = len(self.psi_con.basis_functions)
        state = torch.zeros(state_dim, dtype=torch.float64, device=self.device)
        
        # Gaussian-type trial state
        for i in range(state_dim):
            if i < len(alpha_params):
                state[i] = torch.exp(torch.tensor(-0.5 * alpha_params[i]))
            else:
                state[i] = torch.exp(torch.tensor(-0.5))
        
        # Normalize
        norm = torch.sqrt(torch.sum(state**2))
        if norm > 1e-10:
            state = state / norm
            
        return state
    
    def energy_expectation(self, alpha_params):
        """Calculate energy expectation value"""
        state = self.variational_ansatz(alpha_params)
        H_matrix = self.psi_con.construct_matrix()
        
        # ⟨ψ|H|ψ⟩
        energy = torch.dot(state, H_matrix @ state)
        return float(energy.cpu().numpy())
    
    def find_ground_state(self, max_iterations=500):
        """Find ground state using variational optimization"""
        print("\n=== Universe Ground State Search ===")
        
        # Initial parameters
        n_params = min(10, len(self.psi_con.basis_functions))
        initial_alpha = np.random.randn(n_params) * 0.1
        
        best_energy = float('inf')
        
        def objective(params):
            return self.energy_expectation(params)
        
        # Optimization
        result = minimize(objective, initial_alpha, method='BFGS', 
                        options={'maxiter': max_iterations})
        
        best_energy = result.fun
        best_params = result.x
        ground_state = self.variational_ansatz(best_params)
        
        print(f"Ground state energy: {best_energy:.8f}")
        print(f"Optimization converged: {result.success}")
        
        return ground_state, best_energy, best_params

class NKATUniverseSolver:
    """Main solver for NKAT consciousness eigenvalue problem"""
    
    def __init__(self, N_modes=30, N_cutoff=3):
        """Initialize complete solver system"""
        self.N = N_modes
        self.N_cut = N_cutoff
        
        print(f"🧮 NKAT Universe Solver initialized")
        print(f"Modes: {N_modes}, Cutoff: {N_cutoff}")
        print(f"Total basis size: {N_modes * (N_cutoff + 1)}")
        
        # Initialize quantum spacetime cell foundation
        self.qst_cell = QuantumSpacetimeCell()
        
        # Consciousness operator
        self.psi_con = ConsciousnessOperator(N_modes, N_cutoff)
        
        # Ground state solver
        self.ground_state_solver = UniverseGroundState(self.psi_con)
    
    def solve_eigenvalue_problem(self):
        """Solve the full eigenvalue problem for consciousness operator"""
        print("\n=== Consciousness Operator Eigenvalue Problem ===")
        
        # Construct Hamiltonian matrix
        H_matrix = self.psi_con.construct_matrix()
        print(f"Matrix size: {H_matrix.shape[0]}×{H_matrix.shape[1]}")
        
        # Convert to numpy for eigenvalue computation
        H_np = H_matrix.cpu().numpy()
        
        # Solve eigenvalue problem
        print("Computing eigenvalues...")
        start_time = time.time()
        
        eigenvalues, eigenvectors = eigh(H_np)
        
        computation_time = time.time() - start_time
        print(f"Eigenvalue computation completed in {computation_time:.2f} seconds")
        
        # Extract ground state
        ground_state_index = np.argmin(eigenvalues)
        ground_state_energy = eigenvalues[ground_state_index]
        
        print(f"Ground state energy (exact): {ground_state_energy:.8f}")
        if len(eigenvalues) > 1:
            print(f"First excited state energy: {eigenvalues[1]:.8f}")
            print(f"Energy gap: {eigenvalues[1] - eigenvalues[0]:.8f}")
        
        return eigenvalues, eigenvectors, ground_state_energy
    
    def variational_analysis(self):
        """Perform variational analysis for ground state"""
        ground_state_var, energy_var, params_var = self.ground_state_solver.find_ground_state()
        return ground_state_var, energy_var, params_var
    
    def convergence_analysis(self, N_range=[10, 20, 30]):
        """Analyze convergence with increasing mode number"""
        print("\n=== Convergence Analysis ===")
        
        energies = []
        mode_counts = []
        
        for N in tqdm(N_range, desc="Convergence analysis"):
            # Create temporary solver with different N
            temp_solver = NKATUniverseSolver(N_modes=N, N_cutoff=self.N_cut)
            
            # Solve eigenvalue problem
            eigenvals, _, ground_energy = temp_solver.solve_eigenvalue_problem()
            
            energies.append(ground_energy)
            mode_counts.append(N)
            
            print(f"N={N}: Ground state energy = {ground_energy:.8f}")
        
        return mode_counts, energies
    
    def visualize_results(self, eigenvalues, mode_counts=None, energies=None):
        """Visualize eigenvalue spectrum and convergence"""
        plt.figure(figsize=(15, 10))
        
        # Eigenvalue spectrum
        plt.subplot(2, 2, 1)
        plt.plot(eigenvalues[:20], 'bo-', markersize=6)
        plt.xlabel('Eigenvalue Index')
        plt.ylabel('Energy')
        plt.title('Consciousness Operator Eigenvalue Spectrum')
        plt.grid(True, alpha=0.3)
        
        # Energy gap
        plt.subplot(2, 2, 2)
        if len(eigenvalues) > 1:
            energy_gaps = np.diff(eigenvalues[:min(15, len(eigenvalues))])
            plt.plot(energy_gaps, 'ro-', markersize=6)
        plt.xlabel('Level n')
        plt.ylabel('Energy Gap (E_{n+1} - E_n)')
        plt.title('Energy Level Spacing')
        plt.grid(True, alpha=0.3)
        
        # Convergence analysis
        if mode_counts is not None and energies is not None:
            plt.subplot(2, 2, 3)
            plt.plot(mode_counts, energies, 'go-', markersize=8)
            plt.xlabel('Number of Modes N')
            plt.ylabel('Ground State Energy')
            plt.title('Convergence Analysis')
            plt.grid(True, alpha=0.3)
        
        # Ground state energy distribution
        plt.subplot(2, 2, 4)
        plt.hist(eigenvalues[:min(50, len(eigenvalues))], bins=20, alpha=0.7, edgecolor='black')
        plt.xlabel('Energy')
        plt.ylabel('Frequency')
        plt.title('Energy Level Distribution')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('consciousness_eigenvalue_analysis.png', dpi=300, bbox_inches='tight')
        print("📊 Analysis plots saved as 'consciousness_eigenvalue_analysis.png'")
        plt.show()
        
        return True

def main():
    """Main execution function"""
    print("🌌 NKAT Consciousness-Universe Ground State Solver 🌌")
    print("=" * 60)
    
    # Initialize solver
    solver = NKATUniverseSolver(N_modes=30, N_cutoff=3)
    
    # Solve eigenvalue problem
    eigenvalues, eigenvectors, ground_energy_exact = solver.solve_eigenvalue_problem()
    
    # Variational analysis
    ground_state_var, energy_var, params_var = solver.variational_analysis()
    
    # Convergence analysis
    mode_counts, energies = solver.convergence_analysis()
    
    # Compare exact vs variational
    print(f"\n=== Comparison ===")
    print(f"Exact ground state energy:      {ground_energy_exact:.8f}")
    print(f"Variational ground state energy: {energy_var:.8f}")
    print(f"Variational upper bound gap:     {energy_var - ground_energy_exact:.8f}")
    
    # Visualize results
    solver.visualize_results(eigenvalues, mode_counts, energies)
    
    # Universe properties
    print(f"\n=== Universe Ground State Properties ===")
    print(f"🌍 Universe vacuum energy: {ground_energy_exact:.8e}")
    if len(eigenvalues) > 1:
        print(f"🔮 Consciousness gap: {eigenvalues[1] - eigenvalues[0]:.8e}")
    print(f"💫 Casimir-type regularization: {np.sum(eigenvalues[:10]):.8e}")
    
    print("\n🎯 NKAT universe ground state analysis completed!")
    return solver

if __name__ == "__main__":
    main() 