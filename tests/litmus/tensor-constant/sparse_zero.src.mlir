// VERIFY

func @f() -> f32 {
  %const_1 = constant 1: index
  %c = constant sparse<[[0, 0, 0], [1, 1, 0]],  [-5.0, -2.0]> : tensor<4x4x4xf32>
  %unspecified_loc_has_zero = tensor.extract %c[%const_1, %const_1, %const_1] : tensor<4x4x4xf32>
  return %unspecified_loc_has_zero: f32
}
