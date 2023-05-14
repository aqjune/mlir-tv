// VERIFY

func.func @f() -> f32 {
  %i = tensor.empty (): tensor<f32>
  %a1 = arith.constant sparse<[[0], [1], [2]], [1.0, -0.0, -0.0]> : tensor<3xf32>
  %a2 = arith.constant sparse<[[0], [1], [2]], [1.0, 2.0, 3.0]> : tensor<3xf32>
  %o1 = linalg.dot ins(%a1, %a2 : tensor<3xf32>,tensor<3xf32>)
    outs(%i: tensor<f32>) -> tensor<f32>
  %res1 = tensor.extract %o1[] : tensor<f32>
  return %res1 : f32
}
