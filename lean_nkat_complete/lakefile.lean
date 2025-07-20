
import Lake
open Lake DSL

package nkat_bsd_solver_complete {
  -- Complete package configuration
  version := "3.0.0"
  description := "NKAT BSD Conjecture Solver with Complete AI Support"
}

@[default_target]
lean_lib nkat_bsd_solver_complete {
  -- Complete library configuration
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

-- Complete theorem solver
lean_exe complete_theorem_solver {
  root := `CompleteTheoremSolver
}

require mathlib from git "https://github.com/leanprover-community/mathlib4.git" @ "v4.8.0-rc1"
require aesop from git "https://github.com/JLimperg/aesop" @ "v4.8.0-rc1"
