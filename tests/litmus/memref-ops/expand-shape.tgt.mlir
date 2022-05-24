func.func @f(%arg : memref<1x2xf32>) -> tensor<1x1x2xf32> {
  %memref = memref.expand_shape %arg [[0, 1], [2]] : memref<1x2xf32> into memref<1x1x2xf32>
  %ret = bufferization.to_tensor %memref : memref<1x1x2xf32>
  return %ret: tensor<1x1x2xf32>
}
