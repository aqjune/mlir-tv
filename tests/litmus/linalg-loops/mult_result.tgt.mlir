#map = affine_map<(d0) -> (d0)>
module  {
  func @linalg_op_same_out_tensors(%arg0: tensor<?xf32> {linalg.inplaceable = true}, %arg1: tensor<?xf32> {linalg.inplaceable = true}) -> (tensor<?xf32>, tensor<?xf32>) {
    %0:2 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%arg0 : tensor<?xf32>) outs(%arg1, %arg1 : tensor<?xf32>, tensor<?xf32>) attrs =  {__inplace_results_attr__ = ["true", "false"]} {
    ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):  // no predecessors
      linalg.yield %arg2, %arg2 : f32, f32
    } -> (tensor<?xf32>, tensor<?xf32>)
    return %0#0, %0#1 : tensor<?xf32>, tensor<?xf32>
  }
}

