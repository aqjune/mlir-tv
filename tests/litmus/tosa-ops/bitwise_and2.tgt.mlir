func.func @f() -> tensor<2x2xi32> {
  %t1 = "tosa.const"() {value = dense<[[0, 0], [0, 0]]> : tensor<2x2xi32>} : () -> tensor<2x2xi32>
  return %t1: tensor<2x2xi32>
}
