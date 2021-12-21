#map = affine_map<(d0) -> (d0)>
module  {
  func @unit_dim_for_both_reduction(%arg0: tensor<1x?x1x1xf32>) -> tensor<1x1xf32> {
    %cst = arith.constant 1.000000e+00 : f32
    %0 = tensor.collapse_shape %arg0 [[0, 1, 2, 3]] : tensor<1x?x1x1xf32> into tensor<?xf32>
    %1 = linalg.init_tensor [1] : tensor<1xf32>
    %2 = linalg.fill(%cst, %1) : f32, tensor<1xf32> -> tensor<1xf32> 
    %3 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%0 : tensor<?xf32>) outs(%2 : tensor<1xf32>) {
    ^bb0(%arg1: f32, %arg2: f32):  // no predecessors
      %5 = arith.addf %arg1, %arg2 : f32
      linalg.yield %5 : f32
    } -> tensor<1xf32>
    %4 = tensor.expand_shape %3 [[0, 1]] : tensor<1xf32> into tensor<1x1xf32>
    return %4 : tensor<1x1xf32>
  }
}

