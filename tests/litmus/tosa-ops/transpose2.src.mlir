// VERIFY

func.func @transpose2() -> tensor<3x2xf32> {
    %input = "tosa.const"() {value = dense<[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]> : tensor<2x3xf32>} : () -> tensor<2x3xf32>
    %perms = "tosa.const"() {value = dense<[1, 0]> : tensor<2xi32>} : () -> tensor<2xi32>
    %1 = "tosa.transpose"(%input, %perms) : (tensor<2x3xf32>, tensor<2xi32>) -> tensor<3x2xf32>
    return %1 : tensor<3x2xf32>
}