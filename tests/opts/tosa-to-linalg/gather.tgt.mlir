#map0 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
module  {
  func.func @gather_float(%arg0: tensor<2x3x2xf32>, %arg1: tensor<2x3xi32>) {
    %0 = tensor.empty () : tensor<2x3x2xf32>
    %1 = linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg1 : tensor<2x3xi32>) outs(%0 : tensor<2x3x2xf32>) {
    ^bb0(%arg2: i32, %arg3: f32):  // no predecessors
      %2 = linalg.index 0 : index
      %3 = arith.index_cast %arg2 : i32 to index
      %4 = linalg.index 2 : index
      %5 = tensor.extract %arg0[%2, %3, %4] : tensor<2x3x2xf32>
      linalg.yield %5 : f32
    } -> tensor<2x3x2xf32>
    return
  }
}

