#map0 = affine_map<(d0, d1, d2, d3) -> (d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
module  {
  func @depthwise1(%arg0: tensor<2x5x5x2xf32>, %arg1: tensor<2x2x2x3xf32>, %arg2: tensor<6xf32>) -> tensor<2x4x4x6xf32> {
    %0 = bufferization.to_memref %arg0 : memref<2x5x5x2xf32>
    %1 = bufferization.to_memref %arg1 : memref<2x2x2x3xf32>
    %2 = bufferization.to_memref %arg2 : memref<6xf32>
    %3 = memref.alloc() : memref<2x4x4x2x3xf32>
    %cst = arith.constant 0.000000e+00 : f32
    linalg.fill(%cst, %3) : f32, memref<2x4x4x2x3xf32> 
    %4 = memref.alloc() : memref<2x4x4x6xf32>
    %5 = memref.alloc() : memref<2x4x4x2x3xf32>
    linalg.copy(%3, %5) : memref<2x4x4x2x3xf32>, memref<2x4x4x2x3xf32> 
    linalg.depthwise_conv_2d_nhwc_hwcm {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%0, %1 : memref<2x5x5x2xf32>, memref<2x2x2x3xf32>) outs(%5 : memref<2x4x4x2x3xf32>)
    %6 = memref.collapse_shape %5 [[0], [1], [2], [3, 4]] : memref<2x4x4x2x3xf32> into memref<2x4x4x6xf32>
    %7 = memref.alloc() : memref<2x4x4x6xf32>
    linalg.generic {indexing_maps = [#map0, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2, %6 : memref<6xf32>, memref<2x4x4x6xf32>) outs(%7 : memref<2x4x4x6xf32>) {
    ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):  // no predecessors
      %9 = arith.addf %arg3, %arg4 : f32
      linalg.yield %9 : f32
    }
    %8 = bufferization.to_tensor %7 : memref<2x4x4x6xf32>
    return %8 : tensor<2x4x4x6xf32>
  }
}

