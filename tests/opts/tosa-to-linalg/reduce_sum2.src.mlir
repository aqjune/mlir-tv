// VERIFY-INCORRECT

// Interestingly, this transformation FAILS to verify because filling
// non-identity value (+0.0) to output tensor.

func @f(%t: tensor<3x4x5xf32>) -> tensor<3x1x5xf32> {
  %0 = "tosa.reduce_sum"(%t) {axis = 1 : i64} : (tensor<3x4x5xf32>) -> tensor<3x1x5xf32>
	return %0: tensor<3x1x5xf32>
}
