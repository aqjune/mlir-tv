// VERIFY

func @add(%arg0: tensor<1xf32>, %arg1: tensor<10x9x8x7xf32>) -> tensor<10x9x8x7xf32> {
  %0 = "tosa.add"(%arg0, %arg1) : (tensor<1xf32>, tensor<10x9x8x7xf32>) -> tensor<10x9x8x7xf32>
  return %0 : tensor<10x9x8x7xf32>
}
