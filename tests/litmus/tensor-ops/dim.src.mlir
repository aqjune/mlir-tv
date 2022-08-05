// VERIFY

func @f(%A: tensor<4x?xf32>) -> index {
  %c1 = arith.constant 0 : index
  %y = tensor.dim %A, %c1 : tensor<4x?xf32>
  return %y: index
}
