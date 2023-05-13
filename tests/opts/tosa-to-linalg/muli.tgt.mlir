#map = affine_map<(d0) -> (d0)>
module  {
  func.func @f(%arg0: tensor<8xi32>, %arg1: tensor<8xi32>) -> tensor<8xi32> {
    %0 = tensor.empty () : tensor<8xi32>
    %1 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%arg0, %arg1 : tensor<8xi32>, tensor<8xi32>) outs(%0 : tensor<8xi32>) {
    ^bb0(%arg2: i32, %arg3: i32, %arg4: i32):  // no predecessors
      %2 = arith.muli %arg2, %arg3 : i32
      linalg.yield %2 : i32
    } -> tensor<8xi32>
    return %1 : tensor<8xi32>
  }
}

