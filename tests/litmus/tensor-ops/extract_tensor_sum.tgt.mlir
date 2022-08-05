func @f(%a: tensor<1000xf32>, %b: tensor<1000xf32>) -> tensor<f32> {
  %i = linalg.init_tensor []: tensor<f32>
  %e = linalg.dot ins(%a, %b : tensor<1000xf32>,tensor<1000xf32>)
    outs(%i: tensor<f32>) -> tensor<f32>
  return %e : tensor<f32>
}
