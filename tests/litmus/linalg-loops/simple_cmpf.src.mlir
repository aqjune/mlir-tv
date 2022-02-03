// VERIFY

#accesses = [
  affine_map<(i, j) -> (i, j)>,
  affine_map<(i, j) -> (i, j)>
]
#trait = {
  indexing_maps = #accesses,
  iterator_types = ["parallel", "parallel"]
}
func @dead_linalg_tensor(%arg0 : tensor<7x7xf32>) {
  %0 = linalg.generic #trait ins(%arg0 : tensor<7x7xf32>) outs(%arg0 : tensor<7x7xf32>) {
    ^bb0(%arg1: f32, %arg2: f32):  // no predecessors
      %cst_114 = arith.constant 0.000000e+00 : f32
      %cst_115 = arith.constant 6.000000e+00 : f32
      %269 = arith.cmpf olt, %arg1, %cst_114 : f32
      %270= arith.select %269, %cst_114, %arg1 : f32
      %271 = arith.cmpf olt, %cst_115, %arg1 : f32
      %272= arith.select %271, %cst_115, %270 : f32
      linalg.yield %272 : f32
  } -> tensor<7x7xf32>
  return
}
