// UNSUPPORTED

func @pooling_nhwc_i8_max_tensor(%input: tensor<1x4x4x1xi8>) -> tensor<1x2x2x1xi8> {
  %fake = linalg.init_tensor [3, 3] : tensor<3x3xi8>
  %init = linalg.init_tensor [1, 2, 2, 1] : tensor<1x2x2x1xi8>
  %cst = arith.constant 0 : i8
  %fill = linalg.fill(%cst, %init): i8, tensor<1x2x2x1xi8> -> tensor<1x2x2x1xi8>
  %res = linalg.pooling_nhwc_max {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
    ins(%input, %fake: tensor<1x4x4x1xi8>, tensor<3x3xi8>)
    outs(%fill: tensor<1x2x2x1xi8>) -> tensor<1x2x2x1xi8>
  return %res : tensor<1x2x2x1xi8>
}

// How to reproduce tgt:
// mlir-opt <src>