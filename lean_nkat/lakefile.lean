
import Lake
open Lake DSL

package nkat_bsd_solver {
  -- add package configuration options here
}

@[default_target]
lean_lib nkat_bsd_solver {
  -- add library configuration options here
}

require mathlib from git "https://github.com/leanprover-community/mathlib4.git" @ "v4.8.0-rc1"
