func @buffer_cast(%arg : tensor<0xf32>) -> ()
{
  // This must not be UB.
  %0 = memref.buffer_cast %arg : memref<0xf32>
  return
}
