// EXPECT: "Source is more defined than target"
// SKIP-IDCHECK
// Source is more defined because %c have to point writable memory block in src.
// Pass identity checks since tgt-tgt pair takes too long time.

func.func @f(%a : memref<8x4xf32>, %b: memref<4x2xf32>, %c: memref<8x2xf32>) -> tensor<8x2xf32> {
  %ta = bufferization.to_tensor %a : memref<8x4xf32> to tensor<8x4xf32>
  %tb = bufferization.to_tensor %b : memref<4x2xf32> to tensor<4x2xf32>
  %tc = bufferization.to_tensor %c : memref<8x2xf32> to tensor<8x2xf32>
  %mat = linalg.matmul ins(%ta, %tb: tensor<8x4xf32>, tensor<4x2xf32>) outs(%tc: tensor<8x2xf32>) -> tensor<8x2xf32>
  return %mat : tensor<8x2xf32>
}
