#map = affine_map<(d0) -> (d0)>
module  {
  func.func @generic_mult2(%arg0: tensor<5xf32>, %arg1: tensor<5xf32>,%arg2: tensor<5xf32>) -> (tensor<5xf32>, tensor<5xf32>) {
    return %arg0, %arg2 : tensor<5xf32>, tensor<5xf32>
  }
}

