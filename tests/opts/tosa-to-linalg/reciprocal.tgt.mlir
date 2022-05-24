#map = affine_map<(d0, d1) -> (d0, d1)>
module  {
  func.func @f(%arg0: tensor<10x10xf32>) -> tensor<10x10xf32> {
    %0 = linalg.init_tensor [10, 10] : tensor<10x10xf32>
    %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0 : tensor<10x10xf32>) outs(%0 : tensor<10x10xf32>) {
    ^bb0(%arg1: f32, %arg2: f32):  // no predecessors
      %cst = arith.constant 1.000000e+00 : f32
      %2 = arith.divf %cst, %arg1 : f32
      linalg.yield %2 : f32
    } -> tensor<10x10xf32>
    return %1 : tensor<10x10xf32>
  }
}

