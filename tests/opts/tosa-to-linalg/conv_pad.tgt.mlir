#map0 = affine_map<(d0, d1, d2, d3) -> (d3, d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d3)>
module  {
  func @conv(%arg0: tensor<2x4x4x3xf32>, %arg1: tensor<16x3x6x3xf32>, %arg2: tensor<16xf32>) -> tensor<2x6x9x16xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = linalg.pad_tensor %arg0 low[0, 2, 5, 0] high[0, 2, 5, 0]  {
    ^bb0(%arg3: index, %arg4: index, %arg5: index, %arg6: index):  // no predecessors
      linalg.yield %cst : f32
    } : tensor<2x4x4x3xf32> to tensor<2x8x14x3xf32>
    %cst_0 = arith.constant dense<[1, 2, 3, 0]> : tensor<4xi64>
    %1 = linalg.init_tensor [3, 6, 3, 16] : tensor<3x6x3x16xf32>
    %2 = linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg1 : tensor<16x3x6x3xf32>) outs(%1 : tensor<3x6x3x16xf32>) {
    ^bb0(%arg3: f32, %arg4: f32):  // no predecessors
      linalg.yield %arg3 : f32
    } -> tensor<3x6x3x16xf32>
    %3 = linalg.init_tensor [2, 6, 9, 16] : tensor<2x6x9x16xf32>
    %cst_1 = arith.constant 0.000000e+00 : f32
    %4 = linalg.fill(%cst_1, %3) : f32, tensor<2x6x9x16xf32> -> tensor<2x6x9x16xf32> 
    %5 = linalg.init_tensor [2, 6, 9, 16] : tensor<2x6x9x16xf32>
    %6 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%0, %2 : tensor<2x8x14x3xf32>, tensor<3x6x3x16xf32>) outs(%4 : tensor<2x6x9x16xf32>) -> tensor<2x6x9x16xf32>
    %7 = linalg.generic {indexing_maps = [#map2, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg2, %6 : tensor<16xf32>, tensor<2x6x9x16xf32>) outs(%5 : tensor<2x6x9x16xf32>) {
    ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):  // no predecessors
      %8 = arith.addf %arg3, %arg4 : f32
      linalg.yield %8 : f32
    } -> tensor<2x6x9x16xf32>
    return %7 : tensor<2x6x9x16xf32>
  }
}

