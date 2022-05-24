// VERIFY

func.func @f(%arg : memref<2x3xf32>)
{
  bufferization.clone %arg: memref<2x3xf32> to memref<2x3xf32>
  return
}
