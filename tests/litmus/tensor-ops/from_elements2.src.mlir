// VERIFY

func.func @from_elem() -> tensor<3xf32>
{
  %c1 = arith.constant 3.0 : f32
  %c2 = arith.constant 4.0 : f32
  %c3 = arith.constant 5.0 : f32
  %v = tensor.from_elements %c1, %c2, %c3 : tensor<3xf32>
  return %v : tensor<3xf32>
}
