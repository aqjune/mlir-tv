func.func @f(%arg : memref<2x3xf32>) -> tensor<2x3xf32>
{
  %v = bufferization.clone %arg: memref<2x3xf32> to memref<2x3xf32>
  %0 = bufferization.to_tensor %v : memref<2x3xf32>
  return %0: tensor<2x3xf32>
}
