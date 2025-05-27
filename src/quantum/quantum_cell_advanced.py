import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm, logm
plt.rcParams['font.family'] = 'MS Gothic'

class AdvancedQuantumCell:
    """拡張2ビット量子セルシミュレーション"""
    
    def __init__(self, theta=0.1, lambda_p=1e-35):
        """
        初期化
        theta: 非可換パラメータ
        lambda_p: プランク長（メートル）
        """
        self.theta = theta
        self.lambda_p = lambda_p
        self.hbar = 1.054571817e-34  # プランク定数
        self.c = 299792458  # 光速
        self.G = 6.67430e-11  # 重力定数
        
        # 基底状態の初期化
        self.state = np.zeros(4, dtype=complex)
        self.state[0] = 1.0
        
        # パウリ行列
        self.sigma = {
            'x': np.array([[0, 1], [1, 0]]),
            'y': np.array([[0, -1j], [1j, 0]]),
            'z': np.array([[1, 0], [0, -1]]),
            'i': np.eye(2)
        }

    def geometric_phase(self):
        """Berry位相の計算"""
        X, Y, Z = self.create_space_operators()
        circuit = np.array([X, Y, Z])
        
        # 経路に沿った位相の積分
        phase = 0
        steps = 100
        for t in np.linspace(0, 2*np.pi, steps):
            state = self.state
            next_state = expm(-1j * self.theta * (np.cos(t)*X + np.sin(t)*Y)) @ state
            phase += np.angle(np.vdot(state, next_state))
            self.state = next_state
        
        return phase/(2*np.pi)

    def quantum_metric(self):
        """量子計量テンソルの計算"""
        X, Y, Z = self.create_space_operators()
        operators = [X, Y, Z]
        dim = len(operators)
        g_ij = np.zeros((dim, dim), dtype=complex)
        
        for i in range(dim):
            for j in range(dim):
                # Fubini-Study計量の計算
                g_ij[i,j] = np.trace(operators[i] @ operators[j]) / 4
                
        return g_ij

    def entropy_tensor(self):
        """エントロピーテンソルの計算"""
        rho = np.outer(self.state, self.state.conj())
        S = -np.trace(rho @ logm(rho + np.eye(4)*1e-10))
        
        # エントロピーテンソルの各成分
        X, Y, Z = self.create_space_operators()
        S_tensor = np.zeros((3,3), dtype=complex)
        for i, A in enumerate([X, Y, Z]):
            for j, B in enumerate([X, Y, Z]):
                S_tensor[i,j] = np.trace(rho @ A @ B) * S
        
        return S_tensor

    def create_space_operators(self):
        """拡張空間演算子の生成"""
        X = np.kron(self.sigma['x'], self.sigma['i'])
        Y = np.kron(self.sigma['y'], self.sigma['i'])
        Z = np.kron(self.sigma['z'], self.sigma['i'])
        return X, Y, Z

    def hamiltonian(self, state):
        """ハミルトニアンの作用"""
        X, Y, Z = self.create_space_operators()
        H = (X @ X + Y @ Y + Z @ Z) + self.theta * (X @ Y - Y @ X)
        return -1j * (H @ state)

    def rk4_step(self, state, dt):
        """ルンゲ・クッタ法による1ステップの時間発展"""
        k1 = self.hamiltonian(state)
        k2 = self.hamiltonian(state + dt*k1/2)
        k3 = self.hamiltonian(state + dt*k2/2)
        k4 = self.hamiltonian(state + dt*k3)
        return state + dt*(k1 + 2*k2 + 2*k3 + k4)/6

    def simulate_evolution(self, T=1.0, steps=1000):
        """時間発展のシミュレーション（RK4法）"""
        t = np.linspace(0, T, steps)
        dt = t[1] - t[0]
        
        # 状態の時間発展を記録
        states = np.zeros((steps, 4), dtype=complex)
        states[0] = self.state
        
        # 物理量の記録
        energies = np.zeros(steps)
        entropies = np.zeros(steps)
        berry_phases = np.zeros(steps)
        
        # 初期値の計算
        X, Y, Z = self.create_space_operators()
        H = (X @ X + Y @ Y + Z @ Z) + self.theta * (X @ Y - Y @ X)
        energies[0] = np.real(np.vdot(self.state, H @ self.state))
        rho = np.outer(self.state, self.state.conj())
        entropies[0] = -np.real(np.trace(rho @ logm(rho + np.eye(4)*1e-10)))
        berry_phases[0] = self.geometric_phase()
        
        # 時間発展の計算
        for i in range(1, steps):
            # RK4による状態の更新
            self.state = self.rk4_step(states[i-1], dt)
            states[i] = self.state
            
            # 物理量の計算
            energies[i] = np.real(np.vdot(self.state, H @ self.state))
            rho = np.outer(self.state, self.state.conj())
            entropies[i] = -np.real(np.trace(rho @ logm(rho + np.eye(4)*1e-10)))
            berry_phases[i] = self.geometric_phase()
        
        return {
            'times': t,
            'states': states,
            'energies': energies,
            'entropies': entropies,
            'berry_phases': berry_phases
        }

    def analyze_topology(self):
        """トポロジカル不変量の計算"""
        # 第一チャーン数
        g_ij = self.quantum_metric()
        chern = np.imag(np.trace(g_ij @ g_ij)) / (2*np.pi)
        
        # ベリー曲率
        X, Y, Z = self.create_space_operators()
        F_xy = self.commutator(X, Y)
        berry_curvature = np.trace(F_xy) / (2*np.pi)
        
        return {
            'chern_number': chern,
            'berry_curvature': berry_curvature
        }

    def commutator(self, A, B):
        """交換関係の計算"""
        return A @ B - B @ A

    def plot_results(self, results):
        """結果の可視化"""
        fig = plt.figure(figsize=(15, 10))
        
        # エネルギー
        ax1 = fig.add_subplot(221)
        ax1.plot(results['times'], results['energies'])
        ax1.set_title('エネルギー期待値')
        ax1.set_xlabel('時間')
        ax1.set_ylabel('エネルギー')
        ax1.grid(True)
        
        # エントロピー
        ax2 = fig.add_subplot(222)
        ax2.plot(results['times'], results['entropies'])
        ax2.set_title('フォンノイマンエントロピー')
        ax2.set_xlabel('時間')
        ax2.set_ylabel('エントロピー')
        ax2.grid(True)
        
        # Berry位相
        ax3 = fig.add_subplot(223)
        ax3.plot(results['times'], results['berry_phases'])
        ax3.set_title('Berry位相')
        ax3.set_xlabel('時間')
        ax3.set_ylabel('位相 (2π単位)')
        ax3.grid(True)
        
        # 状態ベクトル
        ax4 = fig.add_subplot(224)
        for i in range(4):
            ax4.plot(results['times'], np.abs(results['states'][:,i])**2, 
                    label=f'|{i//2}{i%2}⟩')
        ax4.set_title('状態ベクトルの確率')
        ax4.set_xlabel('時間')
        ax4.set_ylabel('確率')
        ax4.legend()
        ax4.grid(True)
        
        plt.tight_layout()
        plt.savefig(f'quantum_cell_analysis_theta_{self.theta}.png', dpi=300, bbox_inches='tight')
        plt.close()

def main():
    print("拡張2ビット量子セルシミュレーションを開始します...")
    
    # 異なる非可換パラメータでの解析
    thetas = [0.1, 0.5, 1.0]
    
    for theta in thetas:
        print(f"\n非可換パラメータ θ = {theta} での解析:")
        cell = AdvancedQuantumCell(theta=theta)
        
        # 時間発展のシミュレーション
        results = cell.simulate_evolution(T=2.0, steps=1000)
        cell.plot_results(results)
        print(f"解析結果を保存しました: quantum_cell_analysis_theta_{theta}.png")
        
        # 量子計量の計算
        g_ij = cell.quantum_metric()
        print("\n量子計量テンソル:")
        print(g_ij)
        
        # エントロピーテンソルの計算
        S_tensor = cell.entropy_tensor()
        print("\nエントロピーテンソル:")
        print(S_tensor)
        
        # トポロジカル不変量の計算
        topology = cell.analyze_topology()
        print("\nトポロジカル不変量:")
        print(f"チャーン数: {topology['chern_number']:.4f}")
        print(f"Berry曲率: {topology['berry_curvature']:.4f}")
        
        # 幾何学的位相の計算
        berry_phase = cell.geometric_phase()
        print(f"\nBerry位相: {berry_phase:.4f} × 2π")

if __name__ == "__main__":
    main() 