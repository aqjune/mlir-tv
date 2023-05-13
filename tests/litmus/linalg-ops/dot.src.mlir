// VERIFY

func.func @f(%a: tensor<100xf32>, %b: tensor<100xf32>) -> tensor<f32> {
  %i = tensor.empty (): tensor<f32>
  %zero = arith.constant -0.0 : f32
  %filled = linalg.fill ins(%zero: f32) outs(%i: tensor<f32>) -> tensor<f32>
  %e = linalg.dot ins(%a, %b : tensor<100xf32>,tensor<100xf32>)
    outs(%filled: tensor<f32>) -> tensor<f32>
  return %e : tensor<f32>
}
