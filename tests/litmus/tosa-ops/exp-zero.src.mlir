// VERIFY

func @f() -> tensor<4xf32> {
  %zero = "tosa.const"() {value = dense<-0.0> : tensor<4xf32>} : () -> tensor<4xf32>
  %res = "tosa.exp"(%zero) : (tensor<4xf32>) -> tensor<4xf32>
  return %res: tensor<4xf32>
}
