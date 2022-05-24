func.func @f(%a: tensor<?xf32>, %b: tensor<?xf32>) -> tensor<f32> {
  %i = linalg.init_tensor []: tensor<f32>
  %res = linalg.dot ins(%a, %a : tensor<?xf32>,tensor<?xf32>)
    outs(%i: tensor<f32>) -> tensor<f32>
  return %res : tensor<f32>
}

