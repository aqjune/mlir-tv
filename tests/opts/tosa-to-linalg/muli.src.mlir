// VERIFY

func.func @f(%arg0: tensor<8xi32>, %arg1: tensor<8xi32>) -> tensor<8xi32> {
  %0 = "tosa.mul"(%arg0, %arg1) {shift=0:i8}: (tensor<8xi32>, tensor<8xi32>) -> tensor<8xi32>
  return %0 : tensor<8xi32>
}

