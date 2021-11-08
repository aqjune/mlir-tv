func @f() -> tensor<2x2xi32> {
  %t1 = "tosa.const"() {value = dense<[[-1, -1], [-1, -1]]> : tensor<2x2xi32>} : () -> tensor<2x2xi32>
  return %t1: tensor<2x2xi32>
}
