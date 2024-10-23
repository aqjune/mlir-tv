// VERIFY-INCORRECT

func.func @copy(%m1: memref<10x10xf32>, %m2: memref<10x10xf32>)
{
  %c0 = arith.constant 0.0: f32
  %m1_partial = memref.subview %m1[0,0][9,9][1,1]: memref<10x10xf32> to memref<9x9xf32, strided<[10, 1]>>
  linalg.fill ins(%c0: f32) outs(%m1_partial: memref<9x9xf32, strided<[10, 1]>>)
  memref.copy %m1, %m2 : memref<10x10xf32> to memref<10x10xf32>
  return
}
