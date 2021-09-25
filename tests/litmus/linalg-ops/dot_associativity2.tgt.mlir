func @f() -> f32 {
  %c0 = constant 0 : index
  %i = linalg.init_tensor []: tensor<f32>
  %a1 = constant sparse<[[1], [0]], [-12.0, 3.0]> : tensor<2xf32>
  %a2 = constant sparse<[[2], [1], [0]], [2.0, 5.0, 4.0]> : tensor<3xf32>
  %b1 = constant sparse<[[1], [0]], [1.0, 8.0]> : tensor<2xf32>
  %b2 = constant sparse<[[2], [1], [1]], [5.0, 6.0, 0.0]> : tensor<3xf32>
  %o1 = linalg.dot ins(%a1, %b1 : tensor<2xf32>,tensor<2xf32>)
    outs(%i: tensor<f32>) -> tensor<f32>
  %o2 = linalg.dot ins(%a2, %b2 : tensor<3xf32>,tensor<3xf32>)
    outs(%i: tensor<f32>) -> tensor<f32>
  %res1 = tensor.extract %o1[] : tensor<f32>
  %res2 = tensor.extract %o2[] : tensor<f32>
  %res = addf %res1, %res2 : f32
  return %res : f32
}
