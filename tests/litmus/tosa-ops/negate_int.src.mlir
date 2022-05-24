// VERIFY

func.func @f() -> tensor<4xi32> {
  %c0 = "tosa.const"() {value = dense<[1, 2, 3, 4]> : tensor<4xi32>} : () -> tensor<4xi32>
  %a = "tosa.negate"(%c0) : (tensor<4xi32>) -> tensor<4xi32>
  return %a: tensor<4xi32>
}
