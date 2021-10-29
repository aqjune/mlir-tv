func @f() -> tensor<4xf32> {
  %half = "tosa.const"() {value = dense<[1.0, 2.0]> : tensor<2xf32>} : () -> tensor<2xf32>
  %t = "tosa.concat"(%half, %half) { axis = 0: i64}: (tensor<2xf32>,tensor<2xf32>)->tensor<4xf32>
	return %t: tensor<4xf32>
}
