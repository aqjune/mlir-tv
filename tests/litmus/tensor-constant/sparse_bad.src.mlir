// VERIFY-INCORRECT

func @f() -> f32 {
  %const_5 = arith.constant 5 : index
  %0 = arith.constant sparse<[[5, 5], [2, 3]], [-12.0, 3.0]> : tensor<10x10xf32>
  %minus_twelve = tensor.extract %0[%const_5, %const_5] : tensor<10x10xf32>
  return %minus_twelve : f32
}
