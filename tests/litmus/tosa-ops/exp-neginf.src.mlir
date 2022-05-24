// VERIFY

func.func @f() -> tensor<4xf32> {
  %inf = "tosa.const"() {value = dense<0xFF800000> : tensor<4xf32>} : () -> tensor<4xf32>
  %res = "tosa.exp"(%inf) : (tensor<4xf32>) -> tensor<4xf32>
  return %res: tensor<4xf32>
}
