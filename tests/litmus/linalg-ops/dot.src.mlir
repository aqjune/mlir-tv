// UNSUPPORTED

// This should be verified as correct.

func @f(%a: tensor<?xf32>, %b: tensor<?xf32>) -> tensor<f32> {
  %i = linalg.init_tensor []: tensor<f32>
  %e = linalg.dot ins(%a, %b : tensor<?xf32>,tensor<?xf32>)
    outs(%i: tensor<f32>) -> tensor<f32>
  return %e : tensor<f32>
}
