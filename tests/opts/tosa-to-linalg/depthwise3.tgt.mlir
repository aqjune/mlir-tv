#map0 = affine_map<(d0, d1, d2, d3) -> (d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
module  {
  func @depthwise_conv_quant(%arg0: tensor<1x12x12x4xf32>, %arg1: tensor<3x3x4x128xf32>, %arg2: tensor<512xf32>) {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = linalg.pad_tensor %arg0 low[0, 1, 1, 0] high[0, 1, 1, 0]  {
    ^bb0(%arg3: index, %arg4: index, %arg5: index, %arg6: index):  // no predecessors
      linalg.yield %cst : f32
    } : tensor<1x12x12x4xf32> to tensor<1x14x14x4xf32>
    %1 = linalg.init_tensor [1, 12, 12, 4, 128] : tensor<1x12x12x4x128xf32>
    %cst_0 = arith.constant 0.000000e+00 : f32
    %2 = linalg.fill(%cst_0, %1) : f32, tensor<1x12x12x4x128xf32> -> tensor<1x12x12x4x128xf32> 
    %3 = linalg.init_tensor [1, 12, 12, 512] : tensor<1x12x12x512xf32>
    %4 = linalg.depthwise_conv_2d_nhwc_hwcm {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%0, %arg1 : tensor<1x14x14x4xf32>, tensor<3x3x4x128xf32>) outs(%2 : tensor<1x12x12x4x128xf32>) -> tensor<1x12x12x4x128xf32>
    %5 = tensor.collapse_shape %4 [[0], [1], [2], [3, 4]] : tensor<1x12x12x4x128xf32> into tensor<1x12x12x512xf32>
    %6 = linalg.generic {indexing_maps = [#map0, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg2, %5 : tensor<512xf32>, tensor<1x12x12x512xf32>) outs(%3 : tensor<1x12x12x512xf32>) {
    ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):  // no predecessors
      %7 = arith.addf %arg3, %arg4 : f32
      linalg.yield %7 : f32
    } -> tensor<1x12x12x512xf32>
    return
  }
}

