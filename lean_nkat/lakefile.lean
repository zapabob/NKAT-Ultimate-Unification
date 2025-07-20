
import Lake
open Lake DSL

package nkat_minimal {
  -- 最小限のパッケージ設定
}

@[default_target]
lean_lib nkat_minimal {
  -- 最小限のライブラリ設定
}

@[default_target]
lean_exe nkat_minimal_exe {
  root := `Main
}
