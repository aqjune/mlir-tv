// VERIFY

func @add(%arg0: tensor<10x1xf32>, %arg1: tensor<1x11xf32>) -> tensor<10x11xf32> {
  %0 = "tosa.add"(%arg0, %arg1) : (tensor<10x1xf32>, tensor<1x11xf32>) -> tensor<10x11xf32>
  return %0 : tensor<10x11xf32>
}