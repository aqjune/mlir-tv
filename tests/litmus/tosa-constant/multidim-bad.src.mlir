// VERIFY-INCORRECT

func @f() -> tensor<3x2xi32> {
  %t = "tosa.const"() {value = dense<[[1, 2], [3, 4], [5, 6]]> : tensor<3x2xi32>} : () -> tensor<3x2xi32>
	return %t: tensor<3x2xi32>
}
