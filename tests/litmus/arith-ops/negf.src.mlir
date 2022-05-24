// VERIFY

func.func @f() -> tensor<4xf32> {
  %c0 = "tosa.const"() {value = dense<[1.0, 2.0, 3.0, 4.0]> : tensor<4xf32>} : () -> tensor<4xf32>
  %a = "arith.negf"(%c0) : (tensor<4xf32>) -> tensor<4xf32>
  return %a: tensor<4xf32>
}
