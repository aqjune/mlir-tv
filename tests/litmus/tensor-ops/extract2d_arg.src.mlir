// VERIFY

func.func @extract() -> tensor<2x2xf32>
{
  %c0 = arith.constant 0: index
  %c1 = arith.constant 1: index
  %c2 = arith.constant 2: index
  %v = arith.constant sparse<[[0,0],[0,1],[0,2],[1,0],[1,1],[1,2],[2,0],[2,1],[2,2]], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]> : tensor<3x3xf32>
  %s = tensor.extract_slice %v[%c0,%c1][2,2][%c1,1]: tensor<3x3xf32> to tensor<2x2xf32>
  return %s: tensor<2x2xf32>
}
