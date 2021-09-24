// VERIFY

func @from_elem(%x:f32) -> tensor<3xf32>
{
  %v = tensor.from_elements %x, %x, %x : tensor<3xf32>
  return %v : tensor<3xf32>
}
