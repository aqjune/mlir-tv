// VERIFY

func @depthwise2(%arg0: tensor<1x11x9x3xf32>, %arg1: tensor<3x1x3x11xf32>) -> tensor<1x5x5x3x11xf32> {
  %0 = linalg.init_tensor [1, 5, 5, 3, 11] : tensor<1x5x5x3x11xf32>
  %cst = arith.constant 0.000000e+00 : f32
  %1 = linalg.fill(%cst, %0) : f32, tensor<1x5x5x3x11xf32> -> tensor<1x5x5x3x11xf32> 
  %2 = linalg.depthwise_conv_2d_nhwc_hwcm {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>} ins(%arg0, %arg1 : tensor<1x11x9x3xf32>, tensor<3x1x3x11xf32>) outs(%1 : tensor<1x5x5x3x11xf32>) -> tensor<1x5x5x3x11xf32>
  return %2 : tensor<1x5x5x3x11xf32>
}

// mlir-opt -linalg-bufferize
