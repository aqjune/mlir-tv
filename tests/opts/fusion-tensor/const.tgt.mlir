#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
module  {
  func @generic_op_constant_fusion(%arg0: tensor<5x?x?xf32>) -> tensor<5x?x?xf32> {
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %cst = arith.constant 4.200000e+01 : f32
    %0 = tensor.dim %arg0, %c1 : tensor<5x?x?xf32>
    %1 = tensor.dim %arg0, %c2 : tensor<5x?x?xf32>
    %2 = linalg.init_tensor [5, %0, %1] : tensor<5x?x?xf32>
    %3 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg0 : tensor<5x?x?xf32>) outs(%2 : tensor<5x?x?xf32>) {
    ^bb0(%arg1: f32, %arg2: f32):  // no predecessors
      %4 = arith.mulf %cst, %arg1 : f32
      linalg.yield %4 : f32
    } -> tensor<5x?x?xf32>
    return %3 : tensor<5x?x?xf32>
  }
}

