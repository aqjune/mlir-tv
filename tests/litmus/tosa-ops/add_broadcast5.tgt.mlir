func.func @add(%arg0: tensor<i32>, %arg1: tensor<10x9x8x7xi32>) -> tensor<10x9x8x7xi32> {
  %0 = "tosa.add"(%arg0, %arg1) : (tensor<i32>, tensor<10x9x8x7xi32>) -> tensor<10x9x8x7xi32>
  return %0 : tensor<10x9x8x7xi32>
}