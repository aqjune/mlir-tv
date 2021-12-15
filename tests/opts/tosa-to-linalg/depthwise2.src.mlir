// VERIFY

func @depthwise_conv_strides(%arg0 : tensor<1x11x9x3xf32>, %arg1 : tensor<3x1x3x11xf32>, %arg2 : tensor<33xf32>) -> () {
  %2 = "tosa.depthwise_conv2d"(%arg0, %arg1, %arg2) { pad = [0, 0, 0, 0], stride = [2, 2], dilation = [1, 1] } : (tensor<1x11x9x3xf32>, tensor<3x1x3x11xf32>, tensor<33xf32>)  -> (tensor<1x5x5x33xf32>)
  return
}