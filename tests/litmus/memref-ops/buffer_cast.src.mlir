// VERIFY

func @buffer_cast(%arg : tensor<2x3xf32>) -> f32
{
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = memref.buffer_cast %arg : memref<2x3xf32>
  %1 = memref.load %0[%c0, %c1] : memref<2x3xf32>
  return %1 : f32
}
