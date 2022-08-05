// VERIFY
// ARGS: --use-neg-zero

// Without --use-neg-zero, this transformation is incorrect because tgt is filling
// a non-identity value (+0.0) to the output tensor.

func @f(%t: tensor<3x1000x5xf32>) -> tensor<3x1x5xf32> {
  %0 = "tosa.reduce_sum"(%t) {axis = 1 : i64} : (tensor<3x1000x5xf32>) -> tensor<3x1x5xf32>
	return %0: tensor<3x1x5xf32>
}
