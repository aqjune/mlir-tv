// UNSUPPORTED

func.func @pooling_nhwc_i8_max_tensor(%input: tensor<1x4x4x1xi8>) -> tensor<1x2x2x1xi8> {
  %fake = tensor.empty () : tensor<3x3xi8>
  %init = tensor.empty () : tensor<1x2x2x1xi8>
  %cst = arith.constant 0 : i8
  %fill = linalg.fill ins(%cst: i8) outs(%init: tensor<1x2x2x1xi8>) -> tensor<1x2x2x1xi8>
  %res = linalg.pooling_nhwc_max {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
    ins(%input, %fake: tensor<1x4x4x1xi8>, tensor<3x3xi8>)
    outs(%fill: tensor<1x2x2x1xi8>) -> tensor<1x2x2x1xi8>
  return %res : tensor<1x2x2x1xi8>
}

// How to reproduce tgt:
// mlir-opt <src>