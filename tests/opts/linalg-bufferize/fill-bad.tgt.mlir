func.func @bufferize_fill(%arg0: tensor<?xf32>) -> tensor<?xf32> {
  %0 = bufferization.to_memref %arg0 : memref<?xf32>
  %cst = arith.constant 1.000000e+00 : f32
  linalg.fill ins(%cst: f32) outs(%0: memref<?xf32>)
  %1 = bufferization.to_tensor %0 : memref<?xf32>
  return %1 : tensor<?xf32>
}
