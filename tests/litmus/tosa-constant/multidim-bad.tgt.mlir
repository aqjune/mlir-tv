func.func @f() -> tensor<3x2xi32> {
  %t1 = "tosa.const"() {value = dense<[[1, 2]]> : tensor<1x2xi32>} : () -> tensor<1x2xi32>
  %t2 = "tosa.const"() {value = dense<[[3, 4]]> : tensor<1x2xi32>} : () -> tensor<1x2xi32>
  %t3 = "tosa.const"() {value = dense<[[5, 6]]> : tensor<1x2xi32>} : () -> tensor<1x2xi32>
  %t = "tosa.concat"(%t1, %t3, %t2) { axis = 0: i64}: (tensor<1x2xi32>,tensor<1x2xi32>,tensor<1x2xi32>)->tensor<3x2xi32>
	return %t: tensor<3x2xi32>
}
