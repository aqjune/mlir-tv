// VERIFY

func.func @f() -> tensor<2x2xi32> {
  %t1 = "tosa.const"() {value = dense<[[1, 2], [3, 4]]> : tensor<2x2xi32>} : () -> tensor<2x2xi32>
  %t2 = "tosa.const"() {value = dense<5> : tensor<i32>} : () -> tensor<i32>
  %rt = "tosa.bitwise_and"(%t1, %t2) : (tensor<2x2xi32>, tensor<i32>) -> tensor<2x2xi32>
  return %rt: tensor<2x2xi32>
}
