// VERIFY
// ARGS: --associative

func @f(%a: tensor<1000xf32>, %b: tensor<1000xf32>) -> tensor<f32> {
  %i = linalg.init_tensor []: tensor<f32>
  %a1 = tensor.extract_slice %a[0][500][1]: tensor<1000xf32> to tensor<500xf32>
  %a2 = tensor.extract_slice %a[500][500][1]: tensor<1000xf32> to tensor<500xf32>
  %b1 = tensor.extract_slice %b[0][500][1]: tensor<1000xf32> to tensor<500xf32>
  %b2 = tensor.extract_slice %b[500][500][1]: tensor<1000xf32> to tensor<500xf32>
  %e1 = linalg.dot ins(%a1, %b1 : tensor<500xf32>,tensor<500xf32>)
    outs(%i: tensor<f32>) -> tensor<f32>
  %e2 = linalg.dot ins(%a2, %b2 : tensor<500xf32>,tensor<500xf32>)
    outs(%i: tensor<f32>) -> tensor<f32>
  %r1 = arith.addf %e1, %e2 : tensor<f32>
  return %r1 : tensor<f32>
}
