#map = affine_map<(d0) -> (d0)>
module  {
  func.func @f(%arg0: tensor<8xf32>, %arg1: tensor<8xf32>) -> tensor<8xf32> {
    %0 = linalg.init_tensor [8] : tensor<8xf32>
    %1 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%arg0, %arg1 : tensor<8xf32>, tensor<8xf32>) outs(%0 : tensor<8xf32>) {
    ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):  // no predecessors
      %2 = arith.mulf %arg2, %arg3 : f32
      linalg.yield %2 : f32
    } -> tensor<8xf32>
    return %1 : tensor<8xf32>
  }
}

