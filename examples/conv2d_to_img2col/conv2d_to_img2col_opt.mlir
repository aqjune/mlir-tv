#map0 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1 + d3, d2 + d4, d5)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>
module  {
  func @conv_16433136(%arg0: tensor<1x16x16x4xf32>, %arg1: tensor<3x3x4x16xf32>, %arg2: tensor<1x14x14x16xf32>) -> tensor<1x14x14x16xf32> {
    %0 = linalg.init_tensor [1, 14, 14, 3, 3, 4] : tensor<1x14x14x3x3x4xf32>
    %1 = linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : tensor<1x16x16x4xf32>) outs(%0 : tensor<1x14x14x3x3x4xf32>) {
    ^bb0(%arg3: f32, %arg4: f32):  // no predecessors
      linalg.yield %arg3 : f32
    } -> tensor<1x14x14x3x3x4xf32>
    %2 = linalg.tensor_reshape %1 [[0, 1, 2], [3, 4, 5]] : tensor<1x14x14x3x3x4xf32> into tensor<196x36xf32>
    %3 = linalg.tensor_reshape %arg1 [[0, 1, 2], [3]] : tensor<3x3x4x16xf32> into tensor<36x16xf32>
    %4 = linalg.tensor_reshape %arg2 [[0, 1, 2], [3]] : tensor<1x14x14x16xf32> into tensor<196x16xf32>
    %5 = linalg.matmul ins(%2, %3 : tensor<196x36xf32>, tensor<36x16xf32>) outs(%4 : tensor<196x16xf32>) -> tensor<196x16xf32>
    %6 = linalg.tensor_reshape %5 [[0, 1, 2], [3]] : tensor<196x16xf32> into tensor<1x14x14x16xf32>
    return %6 : tensor<1x14x14x16xf32>
  }
}

