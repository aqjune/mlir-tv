// VERIFY

func.func @f(%arg : memref<2x3xf32>) -> tensor<2x3xf32>
{
  %0 = bufferization.to_tensor %arg : memref<2x3xf32> to tensor<2x3xf32>
  %v = bufferization.clone %arg: memref<2x3xf32> to memref<2x3xf32> // Cannot remove this since it makes the block unwritable
  return %0: tensor<2x3xf32>
}
