// VERIFY

func.func @f(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
  %0 = arith.addf %arg0, %arg1 : tensor<f32>
  return %0 : tensor<f32>
}
