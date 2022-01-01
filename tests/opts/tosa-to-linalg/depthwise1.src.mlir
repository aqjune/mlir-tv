// VERIFY
// ARGS: --use-neg-zero

// Without --use-neg-zero, this transformation is incorrect because tgt is filling
// a non-identity value (+0.0) to the output tensor.

func @depthwise1(%arg0: tensor<2x5x5x2xf32>, %arg1: tensor<2x2x2x3xf32>, %arg2: tensor<6xf32>) -> tensor<2x4x4x6xf32> {
  %0 = "tosa.depthwise_conv2d"(%arg0, %arg1, %arg2) {pad = [0, 0, 0, 0], stride = [1, 1], dilation = [1, 1]} : (tensor<2x5x5x2xf32>, tensor<2x2x2x3xf32>, tensor<6xf32>) -> tensor<2x4x4x6xf32>
  return %0 : tensor<2x4x4x6xf32>
}
