// VERIFY
// ARGS: --use-neg-zero

// Without --use-neg-zero, this transformation is incorrect because tgt is filling
// a non-identity value (+0.0) to the output tensor.

func @conv(%img: tensor<1x2x2x2xf32>, %filter: tensor<1x2x2x2xf32>) -> tensor<1x1x1x1xf32> {
    %c0 = arith.constant -0.0 : f32
    %bias = tensor.from_elements %c0: tensor<1xf32>
    %0 = "tosa.conv2d"(%img, %filter, %bias) {dilation = [1, 1], pad = [0, 0, 0, 0], stride = [1, 1]} : (tensor<1x2x2x2xf32>, tensor<1x2x2x2xf32>, tensor<1xf32>) -> tensor<1x1x1x1xf32>
    return %0 : tensor<1x1x1x1xf32>
}
