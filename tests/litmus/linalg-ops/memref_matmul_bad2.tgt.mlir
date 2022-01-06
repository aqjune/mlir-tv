func @f(%a : memref<8x4xf32>, %b: memref<4x16xf32>, %c: memref<8x16xf32>) -> tensor<8x16xf32> {
  %ta = bufferization.to_tensor %a : memref<8x4xf32>
  %tb = bufferization.to_tensor %b : memref<4x16xf32>
  %tc = bufferization.to_tensor %c : memref<8x16xf32>
  %mat = linalg.matmul ins(%ta, %tb: tensor<8x4xf32>, tensor<4x16xf32>) outs(%tc: tensor<8x16xf32>) -> tensor<8x16xf32>
  return %mat : tensor<8x16xf32>
}
