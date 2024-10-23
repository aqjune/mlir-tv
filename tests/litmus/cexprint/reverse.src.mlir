// EXPECT: "Value: (dim: 2, 2) (0, 0) -> 3, (0, 1) -> 4, (1, 0) -> 1, (1, 1) -> 2"

func.func @f() -> tensor<2x2xi32> {
  %t = "tosa.const"() {value = dense<[[1, 2], [3, 4]]> : tensor<2x2xi32>} : () -> tensor<2x2xi32>
  %rt = "tosa.reverse"(%t) {axis = 0 : i32} : (tensor<2x2xi32>) -> tensor<2x2xi32>
  return %rt: tensor<2x2xi32>
}
