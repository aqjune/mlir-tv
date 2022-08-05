// VERIFY

func @broadcast(%arg0: tensor<17x16x15x14xf32>, %arg1: tensor<15x1xf32>) -> tensor<17x16x15x14xf32> {
  %0 = "tosa.add"(%arg0, %arg1) : (tensor<17x16x15x14xf32>, tensor<15x1xf32>) -> tensor<17x16x15x14xf32>
  return %0 : tensor<17x16x15x14xf32>
}
