func.func @f() -> tensor<4x5x?xf32> {
  %0 = tensor.empty () : tensor<4x5x6xf32>
  %1 = tensor.cast %0 : tensor<4x5x6xf32> to tensor<4x5x?xf32>
  return %1 : tensor<4x5x?xf32>
}
