#map0 = affine_map<(d0, d1, d2, d3) -> (d1, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
module  {
  func.func @f(%arg0: tensor<3x3xf32>) -> tensor<6x9xf32> {
    %0 = linalg.init_tensor [2, 3, 1, 3] : tensor<2x3x1x3xf32>
    %1 = linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : tensor<3x3xf32>) outs(%0 : tensor<2x3x1x3xf32>) {
    ^bb0(%arg1: f32, %arg2: f32):  // no predecessors
      linalg.yield %arg1 : f32
    } -> tensor<2x3x1x3xf32>
    %2 = tensor.collapse_shape %1 [[0, 1, 2], [3]] : tensor<2x3x1x3xf32> into tensor<6x3xf32>
    %3 = linalg.init_tensor [1, 6, 3, 3] : tensor<1x6x3x3xf32>
    %4 = linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2 : tensor<6x3xf32>) outs(%3 : tensor<1x6x3x3xf32>) {
    ^bb0(%arg1: f32, %arg2: f32):  // no predecessors
      linalg.yield %arg1 : f32
    } -> tensor<1x6x3x3xf32>
    %5 = tensor.collapse_shape %4 [[0, 1], [2, 3]] : tensor<1x6x3x3xf32> into tensor<6x9xf32>
    return %5 : tensor<6x9xf32>
  }
}

