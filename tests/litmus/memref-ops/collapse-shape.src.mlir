// VERIFY

func.func @f(%arg : memref<8x4x2xf32>) -> tensor<8x8xf32> {
  %tensor = bufferization.to_tensor %arg : memref<8x4x2xf32>
  %ret = tensor.collapse_shape %tensor [[0], [1, 2]] : tensor<8x4x2xf32> into tensor<8x8xf32>
  return %ret: tensor<8x8xf32>
}
