func @f() -> tensor<2x2xi32> {
  %rt = "tosa.const"() {value = dense<[[3, 4], [1, 2]]> : tensor<2x2xi32>} : () -> tensor<2x2xi32>
  return %rt: tensor<2x2xi32>
}
