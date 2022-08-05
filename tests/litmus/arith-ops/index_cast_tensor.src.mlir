// VERIFY

func @index_cast(%tensor: tensor<i16>) -> tensor<i16> {
  %index_tensor = arith.index_cast %tensor : tensor<i16> to tensor<index>
  %res = arith.index_cast %index_tensor: tensor<index> to tensor<i16>
  return %res : tensor<i16>
}

