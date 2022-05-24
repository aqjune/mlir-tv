module  {
  func.func @broadcast(%arg0: tensor<17x16x15x14xf32>, %arg1: tensor<15x1xf32>) -> tensor<17x16x15x14xf32> {
    %0 = "tosa.reshape"(%arg1) {new_shape = [1, 1, 15, 1]} : (tensor<15x1xf32>) -> tensor<1x1x15x1xf32>
    %1 = "tosa.add"(%arg0, %0) : (tensor<17x16x15x14xf32>, tensor<1x1x15x1xf32>) -> tensor<17x16x15x14xf32>
    return %1 : tensor<17x16x15x14xf32>
  }
}