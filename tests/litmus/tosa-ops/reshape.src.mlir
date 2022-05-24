// VERIFY

func.func @f() -> (i32, i32) {
  %t = "tosa.const"() {value = dense<[1, 2, 3, 4]> : tensor<4xi32>} : () -> tensor<4xi32>
  %t2 = "tosa.reshape"(%t) {new_shape = [2, 2]} : (tensor<4xi32>)  -> tensor<2x2xi32>
	%c0 = arith.constant 0: index
	%c1 = arith.constant 1: index
	%two = tensor.extract %t2[%c0,%c1]: tensor<2x2xi32>
	%three = tensor.extract %t2[%c1,%c0]: tensor<2x2xi32>
	return %two, %three: i32, i32
}
