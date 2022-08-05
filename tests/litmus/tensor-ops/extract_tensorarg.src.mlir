// VERIFY

func @extract(%x:tensor<3x3xf32>) -> tensor<3x3xf32>
{
  %s = tensor.extract_slice %x[0,0][3,3][1,1]: tensor<3x3xf32> to tensor<3x3xf32>
  return %s: tensor<3x3xf32>
}
