// VERIFY
// ARGS: --unroll-fp-sum-bound 54
func @conv(%arg0: tensor<2x4x4x3xf32>, %arg1: tensor<16x3x6x3xf32>, %arg2: tensor<16xf32>) -> tensor<2x6x9x16xf32> {
    %0 = "tosa.conv2d"(%arg0, %arg1, %arg2) {dilation = [1, 1], pad = [2, 2, 5, 5], stride = [1, 1]} : (tensor<2x4x4x3xf32>, tensor<16x3x6x3xf32>, tensor<16xf32>) -> tensor<2x6x9x16xf32>
    return %0 : tensor<2x6x9x16xf32>
}
