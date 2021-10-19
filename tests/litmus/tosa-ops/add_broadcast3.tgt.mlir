func @add(%arg0: tensor<2x1xf32>, %arg1: tensor<1x3xf32>) -> tensor<2x3xf32> {
  %0 = "tosa.add"(%arg0, %arg1) : (tensor<2x1xf32>, tensor<1x3xf32>) -> tensor<2x3xf32>
  return %0 : tensor<2x3xf32>
}