// VERIFY-INCORRECT

func.func @bufferize_fill(%arg0: tensor<?xf32>) -> tensor<?xf32> {
  %c0 = arith.constant 0.0 : f32
  %0 = linalg.fill ins(%c0: f32) outs(%arg0: tensor<?xf32>) -> tensor<?xf32>
  return %0 : tensor<?xf32>
}
