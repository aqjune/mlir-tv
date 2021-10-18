func @extract() -> tensor<2x2xf32>
{
  %v = arith.constant sparse<[[0,0],[0,1],[1,0],[1,1]],[2.0, 3.0, 5.0, 6.0]> : tensor<2x2xf32>
  return %v: tensor<2x2xf32>
}
