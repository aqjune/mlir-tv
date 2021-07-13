// VERIFY

func @parallel_loop_with_invariant() {
  %c0 = constant 0 : index
  %c10 = constant 10 : index
  %c1 = constant 1 : index
  %c7 = constant 7 : i32
  %c8 = constant 8 : i32
  scf.parallel (%arg0, %arg1) = (%c0, %c0) to (%c10, %c10) step (%c1, %c1) {
      %v0 = addi %c7, %c8 : i32
      %v3 = addi %arg0, %arg1 : index
  }


  return
}

// How to reproduce tgt:
// mlir-opt -split-input-file -loop-invariant-code-motion <src>
