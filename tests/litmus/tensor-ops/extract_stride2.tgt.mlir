func @extract() -> tensor<3x2xf32>
{
  %v = arith.constant sparse<[[0,0],[0,1],[1,0],[1,1],[2,0],[2,1]],
                        [2.0, 4.0, 6.0, 8.0, 10.0, 12.0]>: tensor<3x2xf32>
  return %v: tensor<3x2xf32>
}
