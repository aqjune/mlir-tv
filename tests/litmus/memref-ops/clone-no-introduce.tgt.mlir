func @f(%arg : memref<2x3xf32>)
{
  // This makes the block unwritable.
  bufferization.clone %arg: memref<2x3xf32> to memref<2x3xf32>
  return
}
