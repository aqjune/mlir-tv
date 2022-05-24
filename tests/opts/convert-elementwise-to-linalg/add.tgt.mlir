#map = affine_map<() -> ()>
func.func @f(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
  %0 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = []} ins(%arg0, %arg1 : tensor<f32>, tensor<f32>) outs(%arg0 : tensor<f32>) {
  ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):  // no predecessors
    %1 = arith.addf %arg2, %arg3 : f32
    linalg.yield %1 : f32
  } -> tensor<f32>
  return %0 : tensor<f32>
}

