// VERIFY

func @f(%a: tensor<100xf32>, %b: tensor<100xf32>) -> tensor<f32> {
  %i = linalg.init_tensor []: tensor<f32>
  %zero = arith.constant -0.0 : f32
  %filled = linalg.fill(%zero, %i): f32, tensor<f32> -> tensor<f32>
  %e = linalg.dot ins(%a, %b : tensor<100xf32>,tensor<100xf32>)
    outs(%filled: tensor<f32>) -> tensor<f32>
  return %e : tensor<f32>
}
