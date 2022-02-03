#map0 = affine_map<(d0, d1, d2, d3) -> (d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
module  {
  func @depthwise2(%arg0: tensor<2x5x5x2xf32>, %arg1: tensor<2x2x2x3xf32>, %arg2: tensor<6xf32>) -> tensor<2x6x6x6xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.pad %arg0 low[0, 1, 1, 0] high[0, 1, 1, 0]  {
    ^bb0(%arg3: index, %arg4: index, %arg5: index, %arg6: index):  // no predecessors
      linalg.yield %cst : f32
    } : tensor<2x5x5x2xf32> to tensor<2x7x7x2xf32>
    %1 = linalg.init_tensor [2, 6, 6, 2, 3] : tensor<2x6x6x2x3xf32>
    %cst_0 = arith.constant 0.000000e+00 : f32
    %2 = linalg.fill(%cst_0, %1) : f32, tensor<2x6x6x2x3xf32> -> tensor<2x6x6x2x3xf32> 
    %3 = linalg.init_tensor [2, 6, 6, 6] : tensor<2x6x6x6xf32>
    %4 = linalg.depthwise_conv_2d_nhwc_hwcm {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%0, %arg1 : tensor<2x7x7x2xf32>, tensor<2x2x2x3xf32>) outs(%2 : tensor<2x6x6x2x3xf32>) -> tensor<2x6x6x2x3xf32>
    %5 = tensor.collapse_shape %4 [[0], [1], [2], [3, 4]] : tensor<2x6x6x2x3xf32> into tensor<2x6x6x6xf32>
    %6 = linalg.generic {indexing_maps = [#map0, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg2, %5 : tensor<6xf32>, tensor<2x6x6x6xf32>) outs(%3 : tensor<2x6x6x6xf32>) {
    ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):  // no predecessors
      %7 = arith.addf %arg3, %arg4 : f32
      linalg.yield %7 : f32
    } -> tensor<2x6x6x6xf32>
    return %6 : tensor<2x6x6x6xf32>
  }
}

