#map = affine_map<(d0)[s0, s1] -> (d0 * s0 + s1)>
builtin.module  {
  builtin.func @fold_rank_reducing_subview_with_load(%arg0: memref<?x?x?x?x?x?xf32>, %arg1: index, %arg2: index, %arg3: index, %arg4: index, %arg5: index, %arg6: index, %arg7: index, %arg8: index, %arg9: index, %arg10: index, %arg11: index, %arg12: index, %arg13: index, %arg14: index, %arg15: index, %arg16: index) -> f32 {
    %c0 = constant 0 : index
    %0 = affine.apply #map(%arg13)[%arg7, %arg1]
    %1 = affine.apply #map(%arg14)[%arg8, %arg2]
    %2 = affine.apply #map(%c0)[%arg9, %arg3]
    %3 = affine.apply #map(%arg15)[%arg10, %arg4]
    %4 = affine.apply #map(%arg16)[%arg11, %arg5]
    %5 = affine.apply #map(%c0)[%arg12, %arg6]
    %6 = memref.load %arg0[%0, %1, %2, %3, %4, %5] : memref<?x?x?x?x?x?xf32>
    return %6 : f32
  }
}
