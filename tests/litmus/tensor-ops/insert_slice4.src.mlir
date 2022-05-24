// UNSUPPORTED

func.func @insert_slice4(%arg0 : tensor<?x?xf32>, %arg1 : index, %arg2 : index, %arg3 : tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
{
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %0 = tensor.insert_slice %arg0 into %arg3[%c0, %arg1, %c1] [%c4, %c1, %arg2] [%c1, %c1, %c1] : tensor<?x?xf32> into tensor<?x?x?xf32>
  return %0 : tensor<?x?x?xf32>
}
