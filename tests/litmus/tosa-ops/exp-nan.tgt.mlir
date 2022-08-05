func @f() -> tensor<4xf32> {
  %nan = "tosa.const"() {value = dense<0x7FC00000> : tensor<4xf32>} : () -> tensor<4xf32>
  return %nan: tensor<4xf32>
}
