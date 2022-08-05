// VERIFY

func @insert_slice_stride() -> tensor<4x4xf32> {
  %v = arith.constant sparse<[[0,0],[0,1],[0,2],[0,3],[1,0],[1,1],[1,2],[1,3],[2,0],[2,1],[2,2],[2,3],[3,0],[3,1],[3,2],[3,3]],
                      [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]> : tensor<4x4xf32>
  %v1 = arith.constant sparse<[[0,0],[0,1],[1,0],[1,1]],
                      [100.0, 200.0, 300.0, 400.0]> : tensor<2x2xf32>
  %0 = tensor.insert_slice %v1 into %v[1, 1] [2, 2] [2, 2] : tensor<2x2xf32> into tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}
