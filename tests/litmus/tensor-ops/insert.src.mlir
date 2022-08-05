// VERIFY

func @from_elem(%arg: tensor<1xf32>) -> tensor<1xf32>
{
  %i = arith.constant 0 : index
  %c1 = arith.constant 3.0 : f32
  %ret = tensor.insert %c1 into %arg[%i] : tensor<1xf32>
  return %ret : tensor<1xf32>
}
