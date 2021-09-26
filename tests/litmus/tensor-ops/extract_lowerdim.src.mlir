// VERIFY

func @extract() -> tensor<2xf32>
{
  %v = constant sparse<[[0,0,0],[0,0,1],[0,1,0],[0,1,1],[1,0,0],[1,0,1],[1,1,0],[1,1,1]],
                        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]> : tensor<2x2x2xf32>
  %s = tensor.extract_slice %v[0,0,1][1,2,1][1,1,1]: tensor<2x2x2xf32> to tensor<2xf32>
  return %s: tensor<2xf32>
}
