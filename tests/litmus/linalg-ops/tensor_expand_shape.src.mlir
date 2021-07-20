// VERIFY-INCORRECT

// tensor_expand_shape raises UB if casting to the dest type is infeasible.

func @f(%arg0 : tensor<?xf32>) -> tensor<?x2xf32> {
	%v = linalg.tensor_expand_shape %arg0 [[0, 1]] : tensor<?xf32> into tensor<?x2xf32>
	return %v: tensor<?x2xf32>
}
