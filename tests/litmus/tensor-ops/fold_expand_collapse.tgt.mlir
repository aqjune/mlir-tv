func.func @fold_expand_collapse(%arg0 : tensor<?x4xf32>, %sz:index) -> tensor<?x4xf32>
{
  return %arg0 : tensor<?x4xf32>
}

