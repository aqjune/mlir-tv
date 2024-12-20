func.func @f(%a : memref<8x4xf32>, %b: memref<4x2xf32>, %c: memref<8x2xf32>) -> tensor<8x2xf32> {
  linalg.matmul ins(%a, %b: memref<8x4xf32>, memref<4x2xf32>) outs(%c: memref<8x2xf32>)
  %tc = bufferization.to_tensor %c : memref<8x2xf32> to tensor<8x2xf32>
  return %tc: tensor<8x2xf32>
}
