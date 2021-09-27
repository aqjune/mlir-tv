
func @from_elem(%arg: tensor<2xf32>) -> tensor<2xf32>
{
  %c1 = constant 3.0 : f32
  %v = tensor.from_elements %c1, %c1 : tensor<2xf32>
  return %v : tensor<2xf32>
}
