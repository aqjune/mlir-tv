// VERIFY

func @insert_slice1(%arg0 : tensor<2x5x10x15xf32>, %arg1 : tensor<2x5x10x15xf32>) -> tensor<2x5x10x15xf32> {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %c2 = constant 2 : index
  %c10 = constant 10 : index
  %0 = tensor.insert_slice %arg0 into %arg1[0, %c0, 0, %c0] [%c2, 5, %c10, 15] [1, %c1, 1, 1] : tensor<2x5x10x15xf32> into tensor<2x5x10x15xf32>
  return %0 : tensor<2x5x10x15xf32>
}
