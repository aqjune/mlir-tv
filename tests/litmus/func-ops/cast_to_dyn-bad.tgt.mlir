func.func private @dyn_tensor(%t: tensor<?xi32>) -> f32

func.func @dim_mismatch() -> f32 {
  %zt = arith.constant sparse<[[0], [1], [2]], [0, 0, 0]> : tensor<3xi32>
  %zdt = tensor.cast %zt: tensor<3xi32> to tensor<?xi32>
  %r = func.call @dyn_tensor(%zdt): (tensor<?xi32>) -> f32
  return %r: f32
}
