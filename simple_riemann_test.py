import mpmath

# 50桁精度
mpmath.mp.dps = 50

# リーマンゼロ点
t = "14.134725141734693790457251983562470270784257115699243175685567460149963429809256764949010393171561"

# s = 1/2 + i*t
s = mpmath.mpc(0.5, mpmath.mpf(t))

# ζ(s)計算
zeta = mpmath.zeta(s)
abs_val = abs(zeta)

print("50桁精度リーマンゼロ点検証")
print("="*40)
print(f"t = {t[:30]}...")
print(f"s = 0.5 + {t[:15]}...i")
print(f"|ζ(s)| = {float(abs_val):.2e}")

if abs_val < 1e-45:
    print("✅ ゼロ点確認！Re(s)=1/2")
else:
    print("❓ 非ゼロ")

print("検証完了") 