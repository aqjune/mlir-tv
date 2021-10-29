// VERIFY

func @f() -> tensor<4xi32> {
  %t = "tosa.const"() {value = dense<[1, 1, 1, 1]> : tensor<4xi32>} : () -> tensor<4xi32>
	return %t: tensor<4xi32>
}
