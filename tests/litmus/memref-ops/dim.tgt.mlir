func @f(%A: memref<4x?xf32>) -> index {
  %c1 = constant 4 : index
  return %c1: index
}
