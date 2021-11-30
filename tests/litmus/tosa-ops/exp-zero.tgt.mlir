func @f() -> tensor<4xf32> {
  %one = "tosa.const"() {value = dense<1.0> : tensor<4xf32>} : () -> tensor<4xf32>
  return %one: tensor<4xf32>
}
