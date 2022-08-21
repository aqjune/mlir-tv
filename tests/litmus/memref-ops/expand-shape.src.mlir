// VERIFY

func @f(%arg : memref<1x2xf32>) -> tensor<1x1x2xf32> {
  %tensor = bufferization.to_tensor %arg : memref<1x2xf32>
  %ret = linalg.tensor_expand_shape %tensor [[0, 1], [2]] : tensor<1x2xf32> into tensor<1x1x2xf32>
  return %ret: tensor<1x1x2xf32>
}
