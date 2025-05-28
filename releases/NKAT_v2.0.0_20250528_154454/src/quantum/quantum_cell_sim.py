import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
plt.rcParams['font.family'] = 'MS Gothic'  # 日本語フォントの設定

class QuantumCell:
    """2ビット量子セルのシミュレーションクラス"""
    
    def __init__(self, theta=0.1):
        """
        初期化
        theta: 非可換パラメータ
        """
        # 基底状態: |00⟩, |01⟩, |10⟩, |11⟩
        self.basis_states = 4
        self.theta = theta
        
        # 初期状態を|00⟩に設定
        self.state = np.zeros(4, dtype=complex)
        self.state[0] = 1.0
        
        # パウリ行列の定義
        self.sigma_x = np.array([[0, 1], [1, 0]])
        self.sigma_y = np.array([[0, -1j], [1j, 0]])
        self.sigma_z = np.array([[1, 0], [0, -1]])

    def tensor_product(self, A, B):
        """テンソル積の計算"""
        return np.kron(A, B)

    def create_space_operators(self):
        """空間演算子の生成"""
        X = self.tensor_product(self.sigma_x, np.eye(2))
        Y = self.tensor_product(self.sigma_y, np.eye(2))
        Z = self.tensor_product(self.sigma_z, np.eye(2))
        return X, Y, Z

    def create_time_operator(self):
        """時間演算子の生成"""
        return self.tensor_product(np.eye(2), self.sigma_z)

    def commutator(self, A, B):
        """交換関係[A,B]の計算"""
        return A @ B - B @ A

    def non_commutative_evolution(self, steps=100, dt=0.01):
        """非可換時空における量子セルの時間発展"""
        X, Y, Z = self.create_space_operators()
        T = self.create_time_operator()
        
        # 非可換ハミルトニアンの構築
        H = (X @ X + Y @ Y + Z @ Z) + self.theta * self.commutator(X, Y)
        
        # 時間発展の記録
        evolution = []
        times = np.linspace(0, dt*steps, steps)
        
        state = self.state.copy()
        for t in times:
            # 時間発展演算子
            U = expm(-1j * H * t)
            evolved_state = U @ state
            
            # 確率振幅を記録
            evolution.append(np.abs(evolved_state)**2)
            
        return times, np.array(evolution)

    def plot_evolution(self, times, evolution):
        """時間発展のプロット"""
        plt.figure(figsize=(12, 8))
        labels = ['|00⟩', '|01⟩', '|10⟩', '|11⟩']
        for i in range(4):
            plt.plot(times, evolution[:, i], label=labels[i], linewidth=2)
        
        plt.xlabel('時間 (規格化単位)', fontsize=12)
        plt.ylabel('確率', fontsize=12)
        plt.title('非可換時空における2ビット量子セルの時間発展\n(θ={})'.format(self.theta), 
                 fontsize=14, pad=20)
        plt.legend(fontsize=12)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('quantum_cell_evolution.png', dpi=300, bbox_inches='tight')
        plt.close()

    def analyze_system(self):
        """システムの解析"""
        X, Y, Z = self.create_space_operators()
        T = self.create_time_operator()
        
        # 非可換関係の計算
        commutator_XY = self.commutator(X, Y)
        commutator_XT = self.commutator(X, T)
        
        # エネルギースペクトルの計算
        H = (X @ X + Y @ Y + Z @ Z) + self.theta * self.commutator(X, Y)
        eigenvalues = np.linalg.eigvals(H)
        
        return {
            'XY_commutator': commutator_XY,
            'XT_commutator': commutator_XT,
            'energy_spectrum': eigenvalues
        }

def main():
    print("2ビット量子セルシミュレーションを開始します...")
    
    # 異なる非可換パラメータでのシミュレーション
    thetas = [0.1, 0.5, 1.0]
    
    for theta in thetas:
        print(f"\n非可換パラメータ θ = {theta} でのシミュレーション:")
        cell = QuantumCell(theta=theta)
        
        # 時間発展のシミュレーション
        times, evolution = cell.non_commutative_evolution(steps=200)
        cell.plot_evolution(times, evolution)
        print(f"時間発展のプロットを保存しました: quantum_cell_evolution_theta_{theta}.png")
        
        # システム解析
        analysis = cell.analyze_system()
        
        print("\n解析結果:")
        print("1. エネルギー固有値:")
        for i, e in enumerate(analysis['energy_spectrum']):
            print(f"  E_{i} = {e:.4f}")
        
        print("\n2. [X,Y]の非ゼロ成分の大きさ:")
        nonzero = np.abs(analysis['XY_commutator']) > 1e-10
        if np.any(nonzero):
            print(f"  最大値: {np.max(np.abs(analysis['XY_commutator'])):.4f}")
        else:
            print("  すべての成分が実質的にゼロ")

if __name__ == "__main__":
    main() 