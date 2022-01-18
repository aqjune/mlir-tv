module  {
  func @f(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>, %arg2: tensor<?x?xf32>, %arg3: tensor<?x?xf32>) -> (tensor<?x?xf32>, tensor<?x?xf32>) {
    return %arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>
  }
}

