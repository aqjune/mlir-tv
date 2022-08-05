#map0 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
func @f(%arg0: tensor<16x8xi4>, %arg1: tensor<8x32xi4>, %arg2: tensor<16x32xi4>) -> tensor<16x32xi4> {
  %0 = linalg.generic {
      indexing_maps = [#map0, #map1, #map2],
      iterator_types = ["parallel", "parallel", "reduction"]}
    ins(%arg0, %arg1 : tensor<16x8xi4>, tensor<8x32xi4>)
    outs(%arg2 : tensor<16x32xi4>) {
  ^bb0(%arg3: i4, %arg4: i4, %arg5: i4):  // no predecessors
    %.3 = arith.muli %arg3, %arg4 : i4
    %.4 = arith.addi %arg5, %.3 : i4
    linalg.yield %.4 : i4
  } -> tensor<16x32xi4>
  return %0 : tensor<16x32xi4>
}
