#map = affine_map<(d0) -> (d0)>
module  {
  func @f(%arg0: tensor<1x?x1x1xi32>) -> tensor<1x1xi32> {
    %cst = arith.constant 1 : i32
    %0 = linalg.tensor_collapse_shape %arg0 [[0, 1, 2, 3]] : tensor<1x?x1x1xi32> into tensor<?xi32>
    %1 = linalg.init_tensor [1] : tensor<1xi32>
    %2 = linalg.fill(%cst, %1) : i32, tensor<1xi32> -> tensor<1xi32> 
    %3 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%0 : tensor<?xi32>) outs(%2 : tensor<1xi32>) {
    ^bb0(%arg1: i32, %arg2: i32):  // no predecessors
      %5 = arith.addi %arg1, %arg2 : i32
      linalg.yield %5 : i32
    } -> tensor<1xi32>
    %4 = linalg.tensor_expand_shape %3 [[0, 1]] : tensor<1xi32> into tensor<1x1xi32>
    return %4 : tensor<1x1xi32>
  }
}

