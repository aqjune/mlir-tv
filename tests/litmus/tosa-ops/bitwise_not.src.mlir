// VERIFY

func.func @f() -> tensor<2x2xi32> {
  %t1 = "tosa.const"() {value = dense<[[0, 1], [2, 3]]> : tensor<2x2xi32>} : () -> tensor<2x2xi32>
  %rt = "tosa.bitwise_not"(%t1) : (tensor<2x2xi32>) -> tensor<2x2xi32>
  return %rt: tensor<2x2xi32>
}
