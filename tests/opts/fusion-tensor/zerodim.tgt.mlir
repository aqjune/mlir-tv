#map = affine_map<() -> ()>
module  {
  func.func @add_mul_scalar_fusion(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<f32>) -> tensor<f32> {
    %0 = tensor.empty () : tensor<f32>
    %1 = linalg.generic {indexing_maps = [#map, #map, #map, #map], iterator_types = []} ins(%arg0, %arg1, %arg2 : tensor<f32>, tensor<f32>, tensor<f32>) outs(%0 : tensor<f32>) {
    ^bb0(%arg3: f32, %arg4: f32, %arg5: f32, %arg6: f32):  // no predecessors
      %2 = arith.addf %arg3, %arg4 : f32
      %3 = arith.mulf %2, %arg5 : f32
      linalg.yield %3 : f32
    } -> tensor<f32>
    return %1 : tensor<f32>
  }
}

