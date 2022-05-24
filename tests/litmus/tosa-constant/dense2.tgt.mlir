func.func @f() -> tensor<4xi32> {
  %half = "tosa.const"() {value = dense<[1, 2]> : tensor<2xi32>} : () -> tensor<2xi32>
  %t = "tosa.concat"(%half, %half) { axis = 0: i64}: (tensor<2xi32>,tensor<2xi32>)->tensor<4xi32>
	return %t: tensor<4xi32>
}
