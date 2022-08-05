func @depthwise_conv(%arg0 : tensor<1x1x1x3xf32>, %arg1 : tensor<1x1x3x2xf32>, %arg2 : tensor<6xf32>) -> tensor<1x1x1x2xf32> {
  %in = tensor.extract_slice %arg0[0,0,0,2][1,1,1,1][1,1,1,1]: tensor<1x1x1x3xf32> to tensor<1xf32>
  %fil = tensor.extract_slice %arg1[0,0,2,0][1,1,1,2][1,1,1,1]: tensor<1x1x3x2xf32> to tensor<2xf32>
  %in2 = tensor.expand_shape %in [[0,1,2,3]] : tensor<1xf32> into tensor<1x1x1x1xf32>
  %fil2 = tensor.expand_shape %fil [[0,1,2,3]] : tensor<2xf32> into tensor<1x1x1x2xf32>
  %bias = tensor.extract_slice %arg2[4][2][1]: tensor<6xf32> to tensor<2xf32>
  %filperms = "tosa.const"() {value = dense<[3, 0, 1, 2]> : tensor<4xi64>} : () -> tensor<4xi64>
  %fil3 = "tosa.transpose"(%fil2, %filperms) : (tensor<1x1x1x2xf32>, tensor<4xi64>) -> tensor<2x1x1x1xf32>
  %0 = "tosa.conv2d"(%in2, %fil3, %bias) { pad = [0, 0, 0, 0], stride = [1, 1], dilation = [1, 1] } : (tensor<1x1x1x1xf32>, tensor<2x1x1x1xf32>, tensor<2xf32>)  -> (tensor<1x1x1x2xf32>)
  return %0 : tensor<1x1x1x2xf32>
}