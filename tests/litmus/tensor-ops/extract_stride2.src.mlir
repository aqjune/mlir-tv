// VERIFY

func.func @extract() -> tensor<3x2xf32>
{
  %v = arith.constant sparse<[[0,0],[0,1],[0,2],[0,3],[1,0],[1,1],[1,2],[1,3],[2,0],[2,1],[2,2],[2,3],[3,0],[3,1],[3,2],[3,3]],
                        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 0.0, 0.0, 0.0, 0.0]> : tensor<4x4xf32>
  %s = tensor.extract_slice %v[0,1][3,2][1,2]: tensor<4x4xf32> to tensor<3x2xf32>
  return %s: tensor<3x2xf32>
}
