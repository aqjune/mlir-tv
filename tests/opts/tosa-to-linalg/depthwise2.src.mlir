// VERIFY-INCORRECT

// Interestingly, this transformation FAILS to verify because filling
// non-identity value (+0.0) to output tensor.

func @depthwise2(%arg0: tensor<2x5x5x2xf32>, %arg1: tensor<2x2x2x3xf32>, %arg2: tensor<6xf32>) -> tensor<2x6x6x6xf32> {
  %0 = "tosa.depthwise_conv2d"(%arg0, %arg1, %arg2) {pad = [1, 1, 1, 1], stride = [1, 1], dilation = [1, 1]} : (tensor<2x5x5x2xf32>, tensor<2x2x2x3xf32>, tensor<6xf32>) -> tensor<2x6x6x6xf32>
  return %0 : tensor<2x6x6x6xf32>
}
