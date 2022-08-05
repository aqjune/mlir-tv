// VERIFY

func @depthwise_conv(%arg0 : tensor<1x1x1x3xf32>, %arg1 : tensor<1x1x3x2xf32>, %arg2 : tensor<6xf32>) -> tensor<1x1x1x2xf32> {
  %0 = "tosa.depthwise_conv2d"(%arg0, %arg1, %arg2) { pad = [0, 0, 0, 0], stride = [1, 1], dilation = [1, 1] } : (tensor<1x1x1x3xf32>, tensor<1x1x3x2xf32>, tensor<6xf32>)  -> (tensor<1x1x1x6xf32>)
  %1 = tensor.extract_slice %0[0,0,0,4][1,1,1,2][1,1,1,1]: tensor<1x1x1x6xf32> to tensor<2xf32>
  %2 = tensor.expand_shape %1 [[0,1,2,3]] : tensor<2xf32> into tensor<1x1x1x2xf32>
  return %2 : tensor<1x1x1x2xf32>
}