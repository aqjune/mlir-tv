func.func @tensor_load(%arg : memref<2x3xf32>) -> f32
{
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = bufferization.to_tensor %arg : memref<2x3xf32> to tensor<2x3xf32>
  %1 = tensor.extract %0[%c0, %c1] : tensor<2x3xf32>
  memref.store %1, %arg[%c0, %c1] : memref<2x3xf32>
  return %1 : f32
}
