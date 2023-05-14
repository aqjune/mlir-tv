// VERIFY

func.func @depthwise_conv(%arg0 : tensor<2x7x5x3xf32>, %arg1 : tensor<3x1x3x2xf32>, %arg2 : tensor<6xf32>) -> tensor<2x5x5x2xf32> {
  %0 = "tosa.depthwise_conv2d"(%arg0, %arg1, %arg2) { pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>, dilation = array<i64: 1, 1> } : (tensor<2x7x5x3xf32>, tensor<3x1x3x2xf32>, tensor<6xf32>)  -> (tensor<2x5x5x6xf32>)
  %1 = tensor.extract_slice %0[0,0,0,4][2,5,5,2][1,1,1,1]: tensor<2x5x5x6xf32> to tensor<2x5x5x2xf32>
  return %1 : tensor<2x5x5x2xf32>
}