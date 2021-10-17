func @from_elem(%x: f32) -> tensor<3xf32>
{
  %v = arith.constant sparse<[[0],[1],[2]], [3.0, 4.0, 5.0]> : tensor<3xf32>
  return %v : tensor<3xf32>
}
