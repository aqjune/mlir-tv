func @insert_slice4(%arg0: tensor<?x?xf32>, %arg1: index, %arg2: index, %arg3: tensor<?x?x?xf32>) -> tensor<?x?x?xf32> {
    %0 = tensor.insert_slice %arg0 into %arg3[0, %arg1, 1] [4, 1, %arg2] [1, 1, 1] : tensor<?x?xf32> into tensor<?x?x?xf32>
    return %0 : tensor<?x?x?xf32>
}