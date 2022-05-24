// VERIFY

func.func @f() -> f32 {
  %const_1 = arith.constant 1: index
  %c = arith.constant sparse<[[0, 0, 0], [1, 1, 1]],  [-5.0, -2.0]> : tensor<4x4x4xf32>
  %minus_two = tensor.extract %c[%const_1, %const_1, %const_1] : tensor<4x4x4xf32>
  return %minus_two: f32
}
