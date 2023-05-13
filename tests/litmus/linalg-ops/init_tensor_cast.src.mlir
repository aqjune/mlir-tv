// VERIFY

func.func @f() -> (tensor<4x5x?xf32>) {
  %c6 = arith.constant 6 : index
  %0 = tensor.empty (%c6) : tensor<4x5x?xf32>
  return %0 : tensor<4x5x?xf32>
}
