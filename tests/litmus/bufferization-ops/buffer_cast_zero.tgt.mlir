func.func @buffer_cast(%arg : tensor<0xf32>) -> ()
{
  // This must not be UB.
  %0 = bufferization.to_memref %arg : tensor<0xf32> to memref<0xf32>
  return
}
