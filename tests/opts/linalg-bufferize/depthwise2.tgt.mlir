module  {
  func.func @depthwise2(%arg0: tensor<1x11x9x3xf32>, %arg1: tensor<3x1x3x11xf32>) -> tensor<1x5x5x3x11xf32> {
    %0 = bufferization.to_memref %arg0 : tensor<1x11x9x3xf32> to memref<1x11x9x3xf32>
    %1 = bufferization.to_memref %arg1 : tensor<3x1x3x11xf32> to memref<3x1x3x11xf32>
    %2 = memref.alloc() : memref<1x5x5x3x11xf32>
    %cst = arith.constant 0.000000e+00 : f32
    linalg.fill ins(%cst:f32) outs(%2: memref<1x5x5x3x11xf32>)
    %3 = memref.alloc() : memref<1x5x5x3x11xf32>
    memref.copy %2, %3 : memref<1x5x5x3x11xf32> to memref<1x5x5x3x11xf32> 
    linalg.depthwise_conv_2d_nhwc_hwcm {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>} ins(%0, %1 : memref<1x11x9x3xf32>, memref<3x1x3x11xf32>) outs(%3 : memref<1x5x5x3x11xf32>)
    %4 = bufferization.to_tensor %3 : memref<1x5x5x3x11xf32> to tensor<1x5x5x3x11xf32>
    return %4 : tensor<1x5x5x3x11xf32>
  }
}

