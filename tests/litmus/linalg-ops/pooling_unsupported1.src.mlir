// UNSUPPORTED

func.func @pooling_nhwc_i32_max(%input: memref<1x4x4x1xi32>, %fake: memref<3x3xi32>, %output: memref<1x2x2x1xi32>) {
  linalg.pooling_nhwc_max {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
    ins(%input, %fake: memref<1x4x4x1xi32>, memref<3x3xi32>)
    outs(%output: memref<1x2x2x1xi32>)
  return
}

// How to reproduce tgt:
// mlir-opt <src>