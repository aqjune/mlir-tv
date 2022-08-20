func @fold_two_collapses(%arg0 : tensor<?x?x?xf32>) -> tensor<?xf32>
{
  %f1 = linalg.tensor_collapse_shape %arg0 [[0, 1], [2]]
      : tensor<?x?x?xf32> into tensor<?x?xf32>
  %f2 = linalg.tensor_collapse_shape %f1 [[0, 1]]
      : tensor<?x?xf32> into tensor<?xf32>
  return %f2 : tensor<?xf32>
}

