// VERIFY

func.func @f() -> tensor<2x2xi32> {
  %t1 = "tosa.const"() {value = dense<[[1, 2], [3, 4]]> : tensor<2x2xi32>} : () -> tensor<2x2xi32>
  %t2 = "tosa.const"() {value = dense<[[0, 0], [0, 0]]> : tensor<2x2xi32>} : () -> tensor<2x2xi32>
  %rt = "tosa.bitwise_xor"(%t1, %t2) : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi32>
  return %rt: tensor<2x2xi32>
}
