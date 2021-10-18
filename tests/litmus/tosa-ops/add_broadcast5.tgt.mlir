func @add(%arg0: tensor<i32>, %arg1: tensor<17x16x15x14xi32>) -> tensor<17x16x15x14xi32> {
  //  CHECK-NEXT: reshape
  %0 = "tosa.add"(%arg0, %arg1) : (tensor<i32>, tensor<17x16x15x14xi32>) -> tensor<17x16x15x14xi32>
  return %0 : tensor<17x16x15x14xi32>
}