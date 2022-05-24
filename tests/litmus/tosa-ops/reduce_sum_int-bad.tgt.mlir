func.func @f(%t: tensor<3x3x3xi32>) -> tensor<3x3xi32> {
  %0 = "tosa.reduce_sum"(%t) {axis = 1 : i64} : (tensor<3x3x3xi32>) -> tensor<3x1x3xi32>
  %res = tensor.collapse_shape %0 [[0, 1], [2]] : tensor<3x1x3xi32> into tensor<3x3xi32>
	return %res: tensor<3x3xi32>
}
