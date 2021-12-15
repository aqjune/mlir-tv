// VERIFY

func @depthwise_conv_quant(%arg0 : tensor<1x12x12x4xf32>, %arg1 : tensor<3x3x4x128xf32>, %arg2 : tensor<512xf32>) -> () {
  %0 = "tosa.depthwise_conv2d"(%arg0, %arg1, %arg2) {pad = [1, 1, 1, 1], stride = [1, 1], dilation = [1, 1] } : (tensor<1x12x12x4xf32>, tensor<3x3x4x128xf32>, tensor<512xf32>)  -> tensor<1x12x12x512xf32>
  return
}