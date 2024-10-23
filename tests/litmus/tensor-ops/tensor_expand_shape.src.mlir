// EXPECT: "Source is more defined than target"

// tensor_expand_shape raises UB if casting to the dest type is infeasible.

func.func @f(%arg0 : tensor<?xf32>, %sz: index) -> tensor<?x2xf32> {
	%v = tensor.expand_shape %arg0 [[0, 1]] output_shape [%sz, 2]: tensor<?xf32> into tensor<?x2xf32>
	return %v: tensor<?x2xf32>
}
