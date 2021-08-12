// VERIFY

func @f(%arg0 : index) -> (f32, f32) {
  %const_1 = constant 1 : index
  %0 = constant dense<4.0> : tensor<4xf32>
  %ext_1 = tensor.extract %0[%arg0] : tensor<4xf32>
  %1 = constant sparse<[[0, 0, 0], [1, 1, 1]],  [-5.0, -2.0]> : tensor<4x4x4xf32>
  %ext_2 = tensor.extract %1[%const_1, %const_1, %const_1] : tensor<4x4x4xf32>
  return %ext_1, %ext_2 : f32, f32
}
