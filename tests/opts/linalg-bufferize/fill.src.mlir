// VERIFY-INCORRECT

// Interestingly, this transformation FAILS to verify because filling in a read-only block is UB.
// MLIR Doc says:
//    Note, that mutating the result of the to_memref operation leads to undefined behavior.

func @bufferize_fill(%arg0: tensor<?xf32>) -> tensor<?xf32> {
  %c0 = arith.constant 0.0 : f32
  %0 = linalg.fill(%c0, %arg0): f32, tensor<?xf32> -> tensor<?xf32>
  return %0 : tensor<?xf32>
}

// mlir-opt -linalg-bufferize
