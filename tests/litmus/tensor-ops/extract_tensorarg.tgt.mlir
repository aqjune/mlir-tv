func.func @extract(%x:tensor<3x3xf32>) -> tensor<3x3xf32>
{
  return %x: tensor<3x3xf32>
}
