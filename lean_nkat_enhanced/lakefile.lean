
import Lake
open Lake DSL

package nkat_bsd_solver_enhanced {
  -- Enhanced package configuration
  version := "2.0.0"
  description := "NKAT BSD Conjecture Solver with AI Support"
}

@[default_target]
lean_lib nkat_bsd_solver_enhanced {
  -- Enhanced library configuration
  roots := #[`NKAT]
}

-- AI proof generation support
lean_exe ai_proof_generator {
  root := `AIProofGenerator
}

-- Verification system
lean_exe proof_verifier {
  root := `ProofVerifier
}

require mathlib from git "https://github.com/leanprover-community/mathlib4.git" @ "v4.8.0-rc1"
require aesop from git "https://github.com/JLimperg/aesop" @ "v4.8.0-rc1"
