import Mathlib.Data.Real.Basic
import Mathlib.Algebra.Ring.Basic

-- 簡単なMathlibテスト
theorem simple_arithmetic : (1 : ℝ) + 1 = 2 := by
  norm_num

theorem ring_property (a b : ℝ) : a * b = b * a := by
  exact mul_comm a b

-- 基本的な数学的証明
theorem basic_math : ∀ (x : ℝ), x + 0 = x := by
  intro x
  exact add_zero x

-- 実用的な定理
theorem practical_theorem : (2 : ℝ) * 3 = 6 := by
  norm_num
