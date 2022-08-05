// VERIFY

func @f(%A: memref<4x?xf32>) -> index {
  %c1 = arith.constant 0 : index
  %y = memref.dim %A, %c1 : memref<4x?xf32>
  return %y: index
}
