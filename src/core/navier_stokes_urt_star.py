import cupy as cp
import numpy as np

# --- 高次Moyal積（FFT畳み込み, order指定可） ---
def moyal_star_product(f, g, theta, order=1):
    """
    f, g: cupy配列
    theta: 非可換パラメータ
    order: 1以上で高次補正
    """
    f_hat = cp.fft.fftn(f)
    g_hat = cp.fft.fftn(g)
    shape = f.shape
    kx = cp.fft.fftfreq(shape[0])[:, None, None]
    ky = cp.fft.fftfreq(shape[1])[None, :, None]
    kz = cp.fft.fftfreq(shape[2])[None, None, :]
    phase = 1.0
    if order >= 1:
        phase *= cp.exp(1j * theta * (kx * ky - ky * kx))
    if order >= 2:
        phase *= (1 - 0.5 * (theta**2) * (kx**2 * ky**2))
    # order>2も拡張可
    return cp.real(cp.fft.ifftn(f_hat * g_hat * phase))

# --- Runge-Kutta 4次法 ---
def rk4_step(u, p, rhs_func, dt, *args, **kwargs):
    k1 = rhs_func(u, p, *args, **kwargs)
    k2 = rhs_func(u + 0.5*dt*k1, p, *args, **kwargs)
    k3 = rhs_func(u + 0.5*dt*k2, p, *args, **kwargs)
    k4 = rhs_func(u + dt*k3, p, *args, **kwargs)
    return u + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)

# --- Euler法 ---
def euler_step(u, p, rhs_func, dt, *args, **kwargs):
    return u + dt * rhs_func(u, p, *args, **kwargs)

# --- Crank-Nicolson法（雛形, 半陰的） ---
def crank_nicolson_step(u, p, rhs_func, dt, *args, **kwargs):
    # 半陰的解法の雛形（実装詳細は用途に応じて拡張）
    u_half = u + 0.5 * dt * rhs_func(u, p, *args, **kwargs)
    return u + dt * rhs_func(u_half, p, *args, **kwargs)

# --- URT展開（リーマン零点スペクトル基底） ---
def urt_expand(u, urt_modes, x_grid):
    """
    u: cupy配列, urt_modes: 零点リスト, x_grid: 格子座標
    各モードの係数を返す
    """
    coeffs = [cp.sum(u * cp.exp(-1j * lam * x_grid)) for lam in urt_modes]
    return cp.array(coeffs)

def urt_reconstruct(coeffs, urt_modes, x_grid):
    """
    URTモードから場を再構成
    """
    u_rec = cp.zeros_like(x_grid, dtype=cp.complex64)
    for c, lam in zip(coeffs, urt_modes):
        u_rec += c * cp.exp(1j * lam * x_grid)
    return cp.real(u_rec)

# --- 物理量計算 ---
def calc_energy(u):
    """速度場uの全エネルギー"""
    return float(cp.sum(u**2).get())

def calc_vorticity(u):
    """3D速度場uの渦度ノルム（簡易版）"""
    grad = cp.gradient(u)
    # 3Dベクトル場の場合はcurlを計算
    if isinstance(u, (list, tuple)) and len(u) == 3:
        wx = grad[2][1] - grad[1][2]
        wy = grad[0][2] - grad[2][0]
        wz = grad[1][0] - grad[0][1]
        vort = cp.sqrt(wx**2 + wy**2 + wz**2)
        return float(cp.mean(vort).get())
    else:
        return 0.0 