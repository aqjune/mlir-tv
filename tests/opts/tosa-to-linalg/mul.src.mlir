// VERIFY

func @f(%arg0: tensor<8xf32>, %arg1: tensor<8xf32>) -> tensor<8xf32> {
  %0 = "tosa.mul"(%arg0, %arg1) {shift=0:i32}: (tensor<8xf32>, tensor<8xf32>) -> tensor<8xf32>
  return %0 : tensor<8xf32>
}

