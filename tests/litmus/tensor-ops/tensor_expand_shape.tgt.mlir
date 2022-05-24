func.func @f(%arg0 : tensor<?xf32>) -> tensor<?x2xf32> {
	%dummy = tensor.expand_shape %arg0 [[0, 1]] : tensor<?xf32> into tensor<?x3xf32>
	%v = tensor.expand_shape %arg0 [[0, 1]] : tensor<?xf32> into tensor<?x2xf32>
	return %v: tensor<?x2xf32>
}
