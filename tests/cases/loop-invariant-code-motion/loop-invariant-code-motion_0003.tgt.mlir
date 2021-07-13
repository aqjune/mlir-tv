#map = affine_map<(d0) -> (d0 + 1)>
#set = affine_set<(d0, d1) : (d1 - d0 >= 0)>
module  {
  func @invariant_code_inside_affine_if() {
    %0 = memref.alloc() : memref<10xf32>
    %cst = constant 8.000000e+00 : f32
    affine.for %arg0 = 0 to 10 {
      %1 = affine.apply #map(%arg0)
      affine.if #set(%arg0, %1) {
        %2 = addf %cst, %cst : f32
        affine.store %2, %0[%arg0] : memref<10xf32>
      }
    }
    return
  }
}

