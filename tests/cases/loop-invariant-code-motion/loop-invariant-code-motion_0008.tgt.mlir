module  {
  func @invariant_loop_dialect() {
    %c0 = constant 0 : index
    %c10 = constant 10 : index
    %c1 = constant 1 : index
    %0 = memref.alloc() : memref<10xf32>
    %cst = constant 7.000000e+00 : f32
    %cst_0 = constant 8.000000e+00 : f32
    %1 = addf %cst, %cst_0 : f32
    scf.for %arg0 = %c0 to %c10 step %c1 {
    }
    scf.for %arg0 = %c0 to %c10 step %c1 {
    }
    return
  }
}

