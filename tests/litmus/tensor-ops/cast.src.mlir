// VERIFY

func @f(%x: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %y = tensor.cast %x : tensor<?x?xf32> to tensor<4x?xf32>
  %z = tensor.cast %y : tensor<4x?xf32> to tensor<?x?xf32>
  return %z: tensor<?x?xf32>
}
