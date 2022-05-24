module  {
  func.func @pooling_nhwc_i32_max(%arg0: memref<1x4x4x1xi32>, %arg1: memref<3x3xi32>, %arg2: memref<1x2x2x1xi32>) {
    linalg.pooling_nhwc_max {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%arg0, %arg1 : memref<1x4x4x1xi32>, memref<3x3xi32>) outs(%arg2 : memref<1x2x2x1xi32>)
    return
  }
}

