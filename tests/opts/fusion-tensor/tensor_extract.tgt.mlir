#map = affine_map<(d0, d1) -> (d0, d1)>
module  {
  func @sigmoid_dynamic_dim(%arg0: tensor<?x1xf32>) -> tensor<?x1xf32> {
    %cst = constant 5.000000e-01 : f32
    %c0 = constant 0 : index
    %0 = tensor.dim %arg0, %c0 : tensor<?x1xf32>
    %1 = linalg.init_tensor [%0, 1] : tensor<?x1xf32>
    %2 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0 : tensor<?x1xf32>) outs(%1 : tensor<?x1xf32>) {
    ^bb0(%arg1: f32, %arg2: f32):  // no predecessors
      %3 = mulf %arg1, %cst : f32
      linalg.yield %3 : f32
    } -> tensor<?x1xf32>
    return %2 : tensor<?x1xf32>
  }
}

