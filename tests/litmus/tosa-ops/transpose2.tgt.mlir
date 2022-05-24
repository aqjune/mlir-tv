func.func @transpose2() -> tensor<3x2xf32> {
    %i = "tosa.const"() {value = dense<[[0.0, 3.0], [1.0, 4.0], [2.0, 5.0]]> : tensor<3x2xf32>} : () -> tensor<3x2xf32>
    return %i : tensor<3x2xf32>
}