func @insert_slice2() -> tensor<4x4xf32> {
  %v = constant sparse<[[0,0],[0,1],[0,2],[0,3],[1,0],[1,1],[1,2],[1,3],[2,0],[2,1],[2,2],[2,3],[3,0],[3,1],[3,2],[3,3]],
                      [100.0, 200.0, 3.0, 4.0, 300.0, 400.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]> : tensor<4x4xf32>
  return %v : tensor<4x4xf32>
}
