// VERIFY

func @f() -> tensor<4x4xf32> {
  %v = constant sparse<[[0,0],[0,1],[0,2],[0,3],[1,0],[1,1],[1,2],[1,3],[2,0],[2,1],[2,2],[2,3],[3,0],[3,1],[3,2],[3,3]],
                      [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]> : tensor<4x4xf32>
  %v1 = constant sparse<[[0,0],[1,0]], [100.0, 200.0]> : tensor<2x1xf32>
  %0 = tensor.insert_slice %v1 into %v[0, 1] [2, 1] [1, 1] : tensor<2x1xf32> into tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}
