// VERIFY

func @f(%arg : memref<2x3xf32>)
{
  memref.clone %arg: memref<2x3xf32> to memref<2x3xf32>
  return
}
