// VERIFY-INCORRECT

// Folding reduce_sum of axis = 0 into linalg.generic doing reduction over axis = 1 is incorrect.

func @f(%t: tensor<3x3x3xf32>) -> tensor<3x3xf32> {
  %0 = "tosa.reduce_sum"(%t) {axis = 0 : i64} : (tensor<3x3x3xf32>) -> tensor<1x3x3xf32>
  %res = tensor.collapse_shape %0 [[0, 1], [2]] : tensor<1x3x3xf32> into tensor<3x3xf32>
	return %res: tensor<3x3xf32>
}
