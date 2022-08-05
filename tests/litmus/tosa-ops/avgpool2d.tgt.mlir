#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
func @avgpool(%arg0: tensor<1x13x13x1001xf32>) -> tensor<1x1x1x1001xf32> {
  %cst_194 = arith.constant 0.000000e+00 : f32
  %373 = linalg.init_tensor [1, 1, 1, 1001] : tensor<1x1x1x1001xf32>
  %374 = linalg.fill(%cst_194, %373): f32, tensor<1x1x1x1001xf32> -> tensor<1x1x1x1001xf32>
  %375 = linalg.init_tensor [13, 13] : tensor<13x13xf32>
  %376 = linalg.pooling_nhwc_sum {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%arg0, %375 : tensor<1x13x13x1001xf32>, tensor<13x13xf32>) outs(%374 : tensor<1x1x1x1001xf32>) -> tensor<1x1x1x1001xf32>
  %377 = linalg.init_tensor [1, 1, 1, 1001] : tensor<1x1x1x1001xf32>
  %378 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%376 : tensor<1x1x1x1001xf32>) outs(%377 : tensor<1x1x1x1001xf32>) {
  ^bb0(%arg1: f32, %arg2: f32):  // no predecessors
    %c0_196 = arith.constant 0 : index
    %c1_197 = arith.constant 1 : index
    %c0_198 = arith.constant 0 : index
    %c0_199 = arith.constant 0 : index
    %391 = linalg.index 1 : index
    %392 = linalg.index 2 : index
    %393 = arith.subi %c0_198, %391 : index
    %394 = arith.subi %c0_199, %392 : index
    %c13_200 = arith.constant 13 : index
    %395 = arith.cmpi slt, %c13_200, %c1_197 : index
    %396= arith.select %395, %c1_197, %c13_200 : index
    %c13_201 = arith.constant 13 : index
    %397 = arith.cmpi slt, %c13_201, %c1_197 : index
    %398= arith.select %397, %c1_197, %c13_201 : index
    %399 = arith.muli %396, %398 : index
    %400 = arith.index_cast %399 : index to i32
    %401 = arith.sitofp %400 : i32 to f32
    %402 = arith.divf %arg1, %401 : f32
    linalg.yield %402 : f32
  } -> tensor<1x1x1x1001xf32>
  return %378 : tensor<1x1x1x1001xf32>
}
