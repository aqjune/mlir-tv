func.func @copy(%m1: memref<10x10xf32>, %m2: memref<10x10xf32>)
{
  %c0 = arith.constant 0.0: f32
  linalg.fill ins(%c0: f32) outs(%m1: memref<10x10xf32>)
  linalg.fill ins(%c0: f32) outs(%m2: memref<10x10xf32>)
  return
}

