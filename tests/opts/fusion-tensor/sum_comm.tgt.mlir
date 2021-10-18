#map0 = affine_map<(d0) -> (0, d0)>
#map1 = affine_map<(d0) -> (0)>
module  {
  func @consumer_with_reduction(%arg0: tensor<1x10xf32>, %arg1: tensor<1x10xf32>, %arg2: tensor<1xf32>) -> tensor<1xf32> {
    %0 = linalg.generic {indexing_maps = [#map0, #map0, #map1], iterator_types = ["reduction"]} ins(%arg0, %arg1 : tensor<1x10xf32>, tensor<1x10xf32>) outs(%arg2 : tensor<1xf32>) {
    ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):  // no predecessors
      %1 = arith.addf %arg3, %arg4 : f32 // arg3 + arg4 != arg4 + arg3
      %2 = arith.addf %1, %arg5 : f32
      linalg.yield %2 : f32
    } -> tensor<1xf32>
    return %0 : tensor<1xf32>
  }
}

