// VERIFY

// Removing dealloc is always okay because it makes the whole context more defined

func.func @test(%arg : memref<2x3xf32>)
{
  memref.dealloc %arg: memref<2x3xf32>
  return
}
