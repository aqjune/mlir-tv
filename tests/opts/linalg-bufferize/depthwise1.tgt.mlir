module  {
  func @depthwise1(%arg0: tensor<2x5x5x2xf32>, %arg1: tensor<2x2x2x3xf32>) -> tensor<2x4x4x2x3xf32> {
    %0 = bufferization.to_memref %arg0 : memref<2x5x5x2xf32>
    %1 = bufferization.to_memref %arg1 : memref<2x2x2x3xf32>
    %2 = memref.alloc() : memref<2x4x4x2x3xf32>
    %cst = arith.constant 0.000000e+00 : f32
    linalg.fill(%cst, %2) : f32, memref<2x4x4x2x3xf32> 
    %3 = memref.alloc() : memref<2x4x4x2x3xf32>
    memref.copy %2, %3 : memref<2x4x4x2x3xf32> to memref<2x4x4x2x3xf32> 
    linalg.depthwise_conv_2d_nhwc_hwcm {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%0, %1 : memref<2x5x5x2xf32>, memref<2x2x2x3xf32>) outs(%3 : memref<2x4x4x2x3xf32>)
    %4 = bufferization.to_tensor %3 : memref<2x4x4x2x3xf32>
    return %4 : tensor<2x4x4x2x3xf32>
  }
}

