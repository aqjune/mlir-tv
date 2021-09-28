
func @from_elem(%arg: tensor<1xf32>) -> tensor<1xf32>
{
  %c1 = constant 3.0 : f32
  %v = tensor.from_elements %c1 : tensor<1xf32>
  return %v : tensor<1xf32>
}
