import cupy as cp
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# パラメータ
N = 256  # 格子点数
L = 2 * np.pi
dx = L / N
dt = 1e-3
nu = 0.01
theta = 0.1  # 非可換パラメータ
steps = 1000

# 格子
x = cp.linspace(0, L, N, endpoint=False)
y = cp.linspace(0, L, N, endpoint=False)
X, Y = cp.meshgrid(x, y, indexing='ij')

# 統合特解的初期条件（リーマン零点スペクトルを利用）
def riemann_zero_spectrum(n=5):
    return cp.array([14.134725, 21.022040, 25.010858, 30.424876, 32.935062][:n])

lambdas = riemann_zero_spectrum(3)
omega = cp.zeros((N, N))
for lam in lambdas:
    omega += cp.sin(lam * X / L) * cp.cos(lam * Y / L)
omega /= len(lambdas)

# Moyal積（2次まで）による非可換補正
def moyal_star(f, g, theta):
    fx, fy = cp.gradient(f, dx, axis=(0, 1))
    gx, gy = cp.gradient(g, dx, axis=(0, 1))
    term1 = f * g
    term2 = (1j * theta / 2) * (fx * gy - fy * gx)
    return term1 + term2

# 速度場の計算（ストリーム関数→速度）
def compute_velocity(omega):
    omega_hat = cp.fft.fft2(omega)
    kx = cp.fft.fftfreq(N, d=dx) * 2 * np.pi
    ky = cp.fft.fftfreq(N, d=dx) * 2 * np.pi
    KX, KY = cp.meshgrid(kx, ky, indexing='ij')
    K2 = KX**2 + KY**2
    K2[0, 0] = 1  # ゼロ除算回避
    psi_hat = omega_hat / (-K2)
    psi = cp.fft.ifft2(psi_hat).real
    u = cp.gradient(psi, dx, axis=1)
    v = -cp.gradient(psi, dx, axis=0)
    return u, v

# 2次元ラプラシアン
def laplacian(f, dx):
    fxx = cp.gradient(cp.gradient(f, dx, axis=0), dx, axis=0)
    fyy = cp.gradient(cp.gradient(f, dx, axis=1), dx, axis=1)
    return fxx + fyy

# 時間発展
for step in tqdm(range(steps)):
    u, v = compute_velocity(omega)
    # 非可換Moyal積による非線形項
    nonlinear = moyal_star(u, cp.gradient(omega, dx, axis=1), theta) + moyal_star(v, cp.gradient(omega, dx, axis=0), theta)
    # 拡散項
    lap = laplacian(omega, dx)
    # 時間発展（Euler法）
    omega = omega + dt * (-nonlinear.real + nu * lap)
    # 必要に応じて保存・可視化

# CPUに転送して可視化
omega_cpu = cp.asnumpy(omega)
plt.imshow(omega_cpu, cmap='bwr')
plt.title('Final Vorticity (Noncommutative Navier–Stokes)')
plt.colorbar()
plt.savefig('results_nkat_noncommutative_navier_stokes_final.png')
plt.show() 