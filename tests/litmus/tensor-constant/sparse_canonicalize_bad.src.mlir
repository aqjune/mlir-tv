// VERIFY-INCORRECT

func @f() -> f32 {
  %const_5 = constant 5 : index
  %0 = constant sparse<[[5, 5], [2, 3]], [-12.0, 3.0]> : tensor<10x10xf32>
  %1 = tensor.extract %0[%const_5, %const_5] : tensor<10x10xf32>
  return %1 : f32
}
