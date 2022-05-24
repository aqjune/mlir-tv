// VERIFY
// ARGS: -max-const-tensor-size=1

#map0 = affine_map<(d0, d1, d2, d3) -> (d3, d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
func.func @transpose() -> tensor<1x1x2x2xf32> {
  %cst = arith.constant dense<[[[[1.0, 3.0]]], [[[2.0, 4.0]]]]> : tensor<2x1x1x2xf32>
  %1 = linalg.init_tensor [1, 1, 2, 2] : tensor<1x1x2x2xf32>
  %2 = linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst : tensor<2x1x1x2xf32>) outs(%1 : tensor<1x1x2x2xf32>) {
    ^bb0(%arg1: f32, %arg2: f32):  // no predecessors
      linalg.yield %arg1 : f32
    } -> tensor<1x1x2x2xf32>
  return %2 : tensor<1x1x2x2xf32>
}
