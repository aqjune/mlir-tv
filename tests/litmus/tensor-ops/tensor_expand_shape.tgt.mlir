func.func @f(%arg0 : tensor<?xf32>, %sz:index) -> tensor<?x2xf32> {
	%dummy = tensor.expand_shape %arg0 [[0, 1]] output_shape [%sz, 3] : tensor<?xf32> into tensor<?x3xf32>
	%v = tensor.expand_shape %arg0 [[0, 1]] output_shape [%sz, 3] : tensor<?xf32> into tensor<?x2xf32>
	return %v: tensor<?x2xf32>
}
