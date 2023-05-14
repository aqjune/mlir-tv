#map = affine_map<(d0, d1) -> (d0, d1)>
module  {
  func.func @producer_indexed_consumer_fusion(%arg0: tensor<?x?xi32>, %arg1: tensor<?x?xi32>) -> tensor<?x?xi32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %0 = tensor.dim %arg0, %c0 : tensor<?x?xi32>
    %1 = tensor.dim %arg0, %c1 : tensor<?x?xi32>
    %2 = tensor.empty (%0, %1) : tensor<?x?xi32>
    %3 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg1 : tensor<?x?xi32>, tensor<?x?xi32>) outs(%2 : tensor<?x?xi32>) {
    ^bb0(%arg2: i32, %arg3: i32, %arg4: i32):  // no predecessors
      %4 = arith.addi %arg2, %arg3 : i32
      %5 = linalg.index 0 : index
      %6 = linalg.index 1 : index
      %7 = arith.index_cast %5 : index to i32
      %8 = arith.index_cast %6 : index to i32
      %9 = arith.addi %4, %7 : i32
      %10 = arith.subi %9, %8 : i32
      linalg.yield %10 : i32
    } -> tensor<?x?xi32>
    return %3 : tensor<?x?xi32>
  }
}

