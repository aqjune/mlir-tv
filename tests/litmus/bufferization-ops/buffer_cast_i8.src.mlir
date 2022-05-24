// VERIFY

func.func @buffer_cast(%arg : tensor<2x3xi8>) -> i8
{
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = bufferization.to_memref %arg : memref<2x3xi8>
  %1 = memref.load %0[%c0, %c1] : memref<2x3xi8>
  return %1 : i8
}
