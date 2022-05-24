module  {
  func.func @pooling_nhwc_i8_max_tensor(%arg0: tensor<1x4x4x1xi8>) -> tensor<1x2x2x1xi8> {
    %0 = linalg.init_tensor [3, 3] : tensor<3x3xi8>
    %1 = linalg.init_tensor [1, 2, 2, 1] : tensor<1x2x2x1xi8>
    %c0_i8 = arith.constant 0 : i8
    %2 = linalg.fill ins(%c0_i8: i8) outs(%1: tensor<1x2x2x1xi8>) -> tensor<1x2x2x1xi8> 
    %3 = linalg.pooling_nhwc_max {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%arg0, %0 : tensor<1x4x4x1xi8>, tensor<3x3xi8>) outs(%2 : tensor<1x2x2x1xi8>) -> tensor<1x2x2x1xi8>
    return %3 : tensor<1x2x2x1xi8>
  }
}