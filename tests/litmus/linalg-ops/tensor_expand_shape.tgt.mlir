func @f(%arg0 : tensor<?xf32>) -> tensor<?x2xf32> {
	%dummy = linalg.tensor_expand_shape %arg0 [[0, 1]] : tensor<?xf32> into tensor<?x3xf32>
	%v = linalg.tensor_expand_shape %arg0 [[0, 1]] : tensor<?xf32> into tensor<?x2xf32>
	return %v: tensor<?x2xf32>
}
