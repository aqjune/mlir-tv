// VERIFY

func @f() -> tensor<4xf32> {
  %t = "tosa.const"() {value = dense<[1.0, 2.0, 1.0, 2.0]> : tensor<4xf32>} : () -> tensor<4xf32>
	return %t: tensor<4xf32>
}
