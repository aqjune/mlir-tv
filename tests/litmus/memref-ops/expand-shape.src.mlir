// VERIFY

func.func @f(%arg : memref<1x2xf32>) -> tensor<1x1x2xf32> {
  %tensor = bufferization.to_tensor %arg : memref<1x2xf32>
  %ret = tensor.expand_shape %tensor [[0, 1], [2]] output_shape [1,1,2] : tensor<1x2xf32> into tensor<1x1x2xf32>
  return %ret: tensor<1x1x2xf32>
}
