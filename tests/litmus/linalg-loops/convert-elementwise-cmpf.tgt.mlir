#map = affine_map<() -> ()>
module  {
  func.func @cmpf(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<i1> {
    %0 = linalg.init_tensor [] : tensor<i1>
    %1 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = []} ins(%arg0, %arg1 : tensor<f32>, tensor<f32>) outs(%0 : tensor<i1>) {
    ^bb0(%arg2: f32, %arg3: f32, %arg4: i1):  // no predecessors
      %2 = arith.cmpf olt, %arg2, %arg3 : f32
      linalg.yield %2 : i1
    } -> tensor<i1>
    return %1 : tensor<i1>
  }
}
