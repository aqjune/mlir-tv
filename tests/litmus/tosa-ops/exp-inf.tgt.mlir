func @f() -> tensor<4xf32> {
  %inf = "tosa.const"() {value = dense<0x7F800000> : tensor<4xf32>} : () -> tensor<4xf32>
  return %inf: tensor<4xf32>
}
