// VERIFY-INCORRECT

func @copy(%m1: memref<10x10xf32>, %m2: memref<10x10xf32>)
{
  %t = linalg.init_tensor [9, 9]: tensor<9x9xf32>
  %c0 = arith.constant 0.0: f32
  %zerotensor = linalg.fill (%c0, %t): f32, tensor<9x9xf32> -> tensor<9x9xf32>

  %m1_partial = memref.subview %m1[0,0][9,9][1,1]: memref<10x10xf32> to memref<9x9xf32>
  memref.tensor_store %zerotensor, %m1_partial: memref<9x9xf32>
  memref.copy %m1, %m2 : memref<10x10xf32> to memref<10x10xf32>
  return
}
