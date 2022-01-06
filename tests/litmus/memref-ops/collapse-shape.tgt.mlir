func @f(%arg : memref<8x4x2xf32>) -> tensor<8x8xf32> {
  %memref = memref.collapse_shape %arg [[0], [1, 2]] : memref<8x4x2xf32> into memref<8x8xf32>
  %ret = bufferization.to_tensor %memref : memref<8x8xf32>
  return %ret: tensor<8x8xf32>
}
