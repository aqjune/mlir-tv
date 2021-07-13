#set = affine_set<(d0, d1) : (d1 - d0 >= 0)>
module  {
  func @invariant_affine_nested_if() {
    %0 = memref.alloc() : memref<10xf32>
    %cst = constant 8.000000e+00 : f32
    affine.for %arg0 = 0 to 10 {
      affine.for %arg1 = 0 to 10 {
        affine.if #set(%arg0, %arg0) {
          %1 = addf %cst, %cst : f32
          affine.if #set(%arg0, %arg0) {
            %2 = addf %1, %1 : f32
          }
        }
      }
    }
    return
  }
}

