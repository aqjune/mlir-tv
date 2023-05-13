func.func @copy(%m1: memref<10x10xf32>, %m2: memref<10x10xf32>)
{
  %t = tensor.empty (): tensor<10x10xf32>
  %c0 = arith.constant 0.0: f32
  %zerotensor = linalg.fill ins(%c0: f32) outs(%t: tensor<10x10xf32>) -> tensor<10x10xf32>

  memref.tensor_store %zerotensor, %m1: memref<10x10xf32>
  memref.tensor_store %zerotensor, %m2: memref<10x10xf32>
  return
}

