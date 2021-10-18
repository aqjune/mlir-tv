func @add(%arg0: tensor<64x64x1xf32>, %arg1: tensor<1x17xf32>) -> (tensor<64x64x17xf32> ) {
  %0 = "tosa.add"(%arg0, %arg1) : (tensor<64x64x1xf32>, tensor<1x17xf32>) -> tensor<64x64x17xf32>
  return %0 : tensor<64x64x17xf32>
}