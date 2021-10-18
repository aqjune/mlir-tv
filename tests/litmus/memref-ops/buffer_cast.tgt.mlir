func @buffer_cast(%arg : tensor<2x3xf32>) -> f32
{
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = tensor.extract %arg[%c0, %c1] : tensor<2x3xf32>
  return %0 : f32
}
