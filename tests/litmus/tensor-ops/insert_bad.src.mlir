// VERIFY-INCORRECT

func @from_elem(%arg: tensor<2xf32>) -> tensor<2xf32>
{
  %i = constant 0 : index
  %c1 = constant 3.0 : f32
  %ret = tensor.insert %c1 into %arg[%i] : tensor<2xf32>
  return %ret : tensor<2xf32>
}
