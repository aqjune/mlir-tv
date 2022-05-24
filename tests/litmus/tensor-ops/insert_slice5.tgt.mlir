module  {
  func.func @insert_slice_canonicalize(%arg0: tensor<?x?x?xf32>, %arg1: index, %arg2: index, %arg3: tensor<?x?x?xf32>) -> tensor<?x?x?xf32> {
    %0 = tensor.cast %arg0 : tensor<?x?x?xf32> to tensor<4x1x?xf32>
    %1 = tensor.insert_slice %0 into %arg3[0, %arg1, 1] [4, 1, %arg2] [1, 1, 1] : tensor<4x1x?xf32> into tensor<?x?x?xf32>
    return %1 : tensor<?x?x?xf32>
  }
}
