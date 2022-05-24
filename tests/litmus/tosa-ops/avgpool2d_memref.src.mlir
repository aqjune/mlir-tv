// VERIFY
// ARGS: --use-neg-zero

#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
func.func @avgpool(%arg0: tensor<1x7x7x1280xf32>) -> tensor<1x1x1x1280xf32> {
  %cst_0 = arith.constant 0.000000e+00 : f32
  %c49_i32 = arith.constant 49 : i32
  %507 = linalg.init_tensor [1, 1, 1, 1280] : tensor<1x1x1x1280xf32>
  %508 = linalg.fill ins(%cst_0: f32) outs(%507: tensor<1x1x1x1280xf32>) -> tensor<1x1x1x1280xf32>
  %509 = linalg.init_tensor [7, 7] : tensor<7x7xf32>
  %510 = linalg.pooling_nhwc_sum {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%arg0, %509 : tensor<1x7x7x1280xf32>, tensor<7x7xf32>) outs(%508 : tensor<1x1x1x1280xf32>) -> tensor<1x1x1x1280xf32>
  %511 = linalg.init_tensor [1, 1, 1, 1280] : tensor<1x1x1x1280xf32>
  %512 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%510 : tensor<1x1x1x1280xf32>) outs(%511 : tensor<1x1x1x1280xf32>) {
  ^bb0(%arg1: f32, %arg2: f32):  // no predecessors
    %526 = arith.sitofp %c49_i32 : i32 to f32
    %527 = arith.divf %arg1, %526 : f32
    linalg.yield %527 : f32
  } -> tensor<1x1x1x1280xf32>
  return %512 : tensor<1x1x1x1280xf32>
}
