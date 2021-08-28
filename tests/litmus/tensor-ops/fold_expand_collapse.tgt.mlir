func @fold_expand_collapse(%arg0 : tensor<?x4xf32>) -> tensor<?x4xf32>
{
  return %arg0 : tensor<?x4xf32>
}

