#map = affine_map<() -> ()>
module  {
  func.func @select(%arg0: tensor<i1>, %arg1: tensor<i32>, %arg2: tensor<i32>) -> tensor<i32> {
    %0 = linalg.generic {indexing_maps = [#map, #map, #map, #map], iterator_types = []} ins(%arg0, %arg1, %arg2 : tensor<i1>, tensor<i32>, tensor<i32>) outs(%arg1 : tensor<i32>) {
    ^bb0(%arg3: i1, %arg4: i32, %arg5: i32, %arg6: i32):  // no predecessors
      %1= arith.select %arg3, %arg4, %arg5 : i32
      linalg.yield %1 : i32
    } -> tensor<i32>
    return %0 : tensor<i32>
  }
}
