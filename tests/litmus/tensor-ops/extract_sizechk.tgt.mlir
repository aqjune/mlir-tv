func.func @f(%x:tensor<2xf32>) -> tensor<2xf32>
{
  %s = tensor.extract_slice %x[1][2][1]: tensor<2xf32> to tensor<2xf32>
  return %s: tensor<2xf32>
}

