// VERIFY

func.func @extract() -> tensor<2xf32>
{
  %v = arith.constant sparse<[[0],[1]],
                        [5.0, 7.0]> : tensor<2xf32>
  return %v: tensor<2xf32>
}
