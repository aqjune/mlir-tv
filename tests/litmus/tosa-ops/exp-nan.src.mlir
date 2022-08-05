// VERIFY

func @f() -> tensor<4xf32> {
  %nan = "tosa.const"() {value = dense<0x7FC00000> : tensor<4xf32>} : () -> tensor<4xf32>
  %res = "tosa.exp"(%nan) : (tensor<4xf32>) -> tensor<4xf32>
  return %res: tensor<4xf32>
}
