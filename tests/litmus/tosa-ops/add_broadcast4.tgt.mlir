func.func @add(%arg0: tensor<32x32x1xf32>, %arg1: tensor<1x16xf32>) -> (tensor<32x32x16xf32> ) {
  %0 = "tosa.add"(%arg0, %arg1) : (tensor<32x32x1xf32>, tensor<1x16xf32>) -> tensor<32x32x16xf32>
  return %0 : tensor<32x32x16xf32>
}