// VERIFY
// ARGS: --unroll-fp-sum-bound 36 --use-neg-zero

// This transformation is incorrect because it is filling
// a non-identity value (+0.0) to the output tensor.

func @conv(%img: tensor<1x29x29x4xf32>, %filter: tensor<16x3x3x4xf32>) -> tensor<1x14x14x16xf32> {
    %c0 = arith.constant 0.0 : f32
    %bias = tensor.from_elements %c0,%c0,%c0,%c0,%c0,%c0,%c0,%c0,%c0,%c0,%c0,%c0,%c0,%c0,%c0,%c0: tensor<16xf32>
    %0 = "tosa.conv2d"(%img, %filter, %bias) {dilation = [1, 1], pad = [0, 0, 0, 0], stride = [2, 2]} : (tensor<1x29x29x4xf32>, tensor<16x3x3x4xf32>, tensor<16xf32>) -> tensor<1x14x14x16xf32>
    return %0 : tensor<1x14x14x16xf32>
}
