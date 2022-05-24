// VERIFY

func.func @f() -> tensor<4x8xf32> {
  %one = arith.constant 1: index
  %two = arith.constant 2: index
  %c = arith.constant sparse<[[1, 2], [3, 4]], [1.0, 2.0]> : tensor<4x8xf32>
  return %c: tensor<4x8xf32>
}

// mlir-opt -tensor-constant-bufferize sparse.src.mlir
