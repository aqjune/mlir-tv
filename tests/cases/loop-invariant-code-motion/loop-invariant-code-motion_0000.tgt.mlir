module  {
  func @nested_loops_both_having_invariant_code() {
    %0 = memref.alloc() : memref<10xf32>
    %cst = constant 7.000000e+00 : f32
    %cst_0 = constant 8.000000e+00 : f32
    %1 = addf %cst, %cst_0 : f32
    %2 = addf %1, %cst_0 : f32
    affine.for %arg0 = 0 to 10 {
      affine.for %arg1 = 0 to 10 {
        affine.store %1, %0[%arg0] : memref<10xf32>
      }
    }
    return
  }
}

