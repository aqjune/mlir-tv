// VERIFY

func @fold_two_collapses(%arg0 : tensor<?x?x?xf32>) -> tensor<?xf32>
{
  %f1 = tensor.collapse_shape %arg0 [[0, 1, 2]]
      : tensor<?x?x?xf32> into tensor<?xf32>
  return %f1 : tensor<?xf32>
}

