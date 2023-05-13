// VERIFY

func.func @f(%a: tensor<100xf32>, %b: tensor<100xf32>) -> tensor<f32> {
  %i = tensor.empty (): tensor<f32>
  %e = linalg.dot ins(%a, %b : tensor<100xf32>,tensor<100xf32>)
    outs(%i: tensor<f32>) -> tensor<f32>
  return %e : tensor<f32>
}
