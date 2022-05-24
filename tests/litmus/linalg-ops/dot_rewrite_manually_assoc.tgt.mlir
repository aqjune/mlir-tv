func.func @f() -> tensor<f32> {
  %zero = arith.constant -0.0 : f32
  %i = linalg.init_tensor []: tensor<f32>
  %outty = linalg.fill ins(%zero: f32) outs(%i: tensor<f32>) -> tensor<f32>
  %a = arith.constant sparse<[[4], [3], [2], [1], [0]], [-12.0, 3.0, 2.0, 5.0, 4.0]> : tensor<5xf32>
  %b = arith.constant sparse<[[4], [3], [2], [1], [0]], [1.0, 8.0, 5.0, 6.0, 0.0]> : tensor<5xf32>
  %res = linalg.dot ins(%a, %b : tensor<5xf32>,tensor<5xf32>)
    outs(%outty: tensor<f32>) -> tensor<f32>
  return %res : tensor<f32>
}
