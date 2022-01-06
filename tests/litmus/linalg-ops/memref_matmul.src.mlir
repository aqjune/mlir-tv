// VERIFY
// NO-IDENTITY
// ARGS: --smt-use-all-logic

memref.global "private" constant @constant_a : memref<8x4xf32> = dense<1.0>
memref.global "private" constant @constant_b : memref<4x16xf32> = dense<2.0>
func @f() -> tensor<8x16xf32> {
  %a = memref.get_global @constant_a : memref<8x4xf32>
  %b = memref.get_global @constant_b : memref<4x16xf32>
  %cst = arith.constant -0.0 : f32
  %c = memref.alloc(): memref<8x16xf32>
  linalg.fill(%cst, %c) : f32, memref<8x16xf32>
  linalg.matmul ins(%a, %b: memref<8x4xf32>, memref<4x16xf32>) outs(%c: memref<8x16xf32>)
  %ret = bufferization.to_tensor %c : memref<8x16xf32>
  return %ret: tensor<8x16xf32>
}
