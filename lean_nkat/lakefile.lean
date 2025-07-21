import Lake
open Lake DSL

package nkat_minimal {
  -- 最小限のパッケージ設定
}

-- 安定版のMathlibを使用（4.7.0対応）
require mathlib from git "https://github.com/leanprover-community/mathlib4.git" @ "v4.7.0"

@[default_target]
lean_lib nkat_minimal {
  -- 最小限のライブラリ設定
  roots := #[`nkat_minimal, `simple_mathlib_test]
}

@[default_target]
lean_exe nkat_minimal_exe {
  root := `Main
}
