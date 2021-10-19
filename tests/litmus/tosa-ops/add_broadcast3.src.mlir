// VERIFY

func @add(%arg0: tensor<5x1xf32>, %arg1: tensor<1x6xf32>) -> tensor<5x6xf32> {
  %0 = "tosa.add"(%arg0, %arg1) : (tensor<5x1xf32>, tensor<1x6xf32>) -> tensor<5x6xf32>
  return %0 : tensor<5x6xf32>
}