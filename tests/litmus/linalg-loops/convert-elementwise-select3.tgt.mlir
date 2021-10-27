#map = affine_map<(d0, d1) -> (d0, d1)>
module  {
  func @select(%arg0: tensor<?x?xi1>, %arg1: tensor<?x?xi32>, %arg2: tensor<?x?xi32>) -> tensor<?x?xi32> {
    %0 = linalg.generic {indexing_maps = [#map, #map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg1, %arg2 : tensor<?x?xi1>, tensor<?x?xi32>, tensor<?x?xi32>) outs(%arg1 : tensor<?x?xi32>) {
    ^bb0(%arg3: i1, %arg4: i32, %arg5: i32, %arg6: i32):  // no predecessors
      %1 = select %arg3, %arg4, %arg5 : i32
      linalg.yield %1 : i32
    } -> tensor<?x?xi32>
    return %0 : tensor<?x?xi32>
  }
}

