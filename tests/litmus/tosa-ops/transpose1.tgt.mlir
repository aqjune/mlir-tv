func @transpose1() -> tensor<3x1x4x2xi32> {
    %0 = "tosa.const"() {value = dense<[[[[0, 12], [1, 13], [2, 14], [3, 15]]], [[[4, 16], [5, 17], [6, 18], [7, 19]]], [[[8, 20], [9, 21], [10, 22], [11, 23]]]]> : tensor<3x1x4x2xi32>} : () -> tensor<3x1x4x2xi32>
    return %0 : tensor<3x1x4x2xi32>
}