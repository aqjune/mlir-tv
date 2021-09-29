func @extract() -> tensor<2xf32>
{
  %v = constant sparse<[[1],[2]],[2.0, 3.0]> : tensor<3xf32>
  %s = tensor.extract_slice %v[1][2][1]: tensor<3xf32> to tensor<2xf32>
  return %s: tensor<2xf32>
}
