#map0 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d1, d2)>
module  {
  func @f(%arg0: tensor<3x4x5xf32>) -> tensor<1x4x5xf32> {
    %0 = linalg.init_tensor [4, 5] : tensor<4x5xf32>
    %cst = arith.constant 0.000000e+00 : f32
    %1 = linalg.fill(%cst, %0): f32, tensor<4x5xf32> -> tensor<4x5xf32> 
    %2 = linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["reduction", "parallel", "parallel"]} ins(%arg0 : tensor<3x4x5xf32>) outs(%1 : tensor<4x5xf32>) {
    ^bb0(%arg1: f32, %arg2: f32):  // no predecessors
      %4 = arith.addf %arg1, %arg2 : f32
      linalg.yield %4 : f32
    } -> tensor<4x5xf32>
    %3 = tensor.expand_shape %2 [[0, 1], [2]] : tensor<4x5xf32> into tensor<1x4x5xf32>
    return %3 : tensor<1x4x5xf32>
  }
}

