// VERIFY

func @f() -> tensor<2x2xi32> {
  %t = "tosa.const"() {value = dense<[[1, 2], [3, 4]]> : tensor<2x2xi32>} : () -> tensor<2x2xi32>
  %rt = "tosa.reverse"(%t) {axis = 0 : i64} : (tensor<2x2xi32>) -> tensor<2x2xi32>
  return %rt: tensor<2x2xi32>
}
