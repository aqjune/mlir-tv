// VERIFY

func @variant_loop_dialect() {
  %ci0 = constant 0 : index
  %ci10 = constant 10 : index
  %ci1 = constant 1 : index
  %m = memref.alloc() : memref<10xf32>
  scf.for %arg0 = %ci0 to %ci10 step %ci1 {
    scf.for %arg1 = %ci0 to %ci10 step %ci1 {
      %v0 = addi %arg0, %arg1 : index
    }
  }


  return
}

// How to reproduce tgt:
// mlir-opt -split-input-file -loop-invariant-code-motion <src>
