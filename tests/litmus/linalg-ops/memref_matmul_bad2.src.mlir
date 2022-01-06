// EXPECT: "Memory mismatch"
// NO-IDENTITY
// ARGS: --smt-use-all-logic

// Source writes matmul results in %c, but target does not.

func @f(%a : memref<8x4xf32>, %b: memref<4x16xf32>, %c: memref<8x16xf32>) -> tensor<8x16xf32> {
  linalg.matmul ins(%a, %b: memref<8x4xf32>, memref<4x16xf32>) outs(%c: memref<8x16xf32>)
  %tc = bufferization.to_tensor %c : memref<8x16xf32>
  return %tc: tensor<8x16xf32>
}
