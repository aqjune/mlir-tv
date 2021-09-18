// VERIFY-INCORRECT

func @from_elem(%x: f32) -> tensor<3xf32>
{
  %c1 = constant 3.0 : f32
  %c2 = constant 4.0 : f32
  %v = tensor.from_elements %c1, %c2, %x : tensor<3xf32>
  return %v : tensor<3xf32>
}
