#map = affine_map<(d0, d1) -> (d0, d1)>
module  {
  func @producer_indexed_consumer_fusion(%arg0: tensor<?x?xi32>, %arg1: tensor<?x?xi32>) -> tensor<?x?xi32> {
    %c0 = constant 0 : index
    %c1 = constant 1 : index
    %0 = tensor.dim %arg0, %c0 : tensor<?x?xi32>
    %1 = tensor.dim %arg0, %c1 : tensor<?x?xi32>
    %2 = linalg.init_tensor [%0, %1] : tensor<?x?xi32>
    %3 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg1 : tensor<?x?xi32>, tensor<?x?xi32>) outs(%2 : tensor<?x?xi32>) {
    ^bb0(%arg2: i32, %arg3: i32, %arg4: i32):  // no predecessors
      %4 = addi %arg2, %arg3 : i32
      %5 = linalg.index 0 : index
      %6 = linalg.index 1 : index
      %7 = index_cast %5 : index to i32
      %8 = index_cast %6 : index to i32
      %9 = addi %4, %7 : i32
      %10 = subi %9, %8 : i32
      linalg.yield %10 : i32
    } -> tensor<?x?xi32>
    return %3 : tensor<?x?xi32>
  }
}

