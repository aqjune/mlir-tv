func.func @conv(%arg0: tensor<1x16x16x4xf32>, %arg1: tensor<3x3x4x16xf32>) -> tensor<1x14x14x16xf32> {
    %out = tensor.empty () : tensor<1x16x14x14xf32>
    %inperms = "tosa.const"() {value = dense<[0, 3, 1, 2]> : tensor<4xi32>} : () -> tensor<4xi32>
    %filperms = "tosa.const"() {value = dense<[3, 2, 0, 1]> : tensor<4xi32>} : () -> tensor<4xi32>
    %outperms = "tosa.const"() {value = dense<[0, 2, 3, 1]> : tensor<4xi32>} : () -> tensor<4xi32>
    %arg3 = "tosa.transpose"(%arg0, %inperms) : (tensor<1x16x16x4xf32>, tensor<4xi32>) -> tensor<1x4x16x16xf32>
    %arg4 = "tosa.transpose"(%arg1, %filperms) : (tensor<3x3x4x16xf32>, tensor<4xi32>) -> tensor<16x4x3x3xf32>
    %0 = linalg.conv_2d_nchw_fchw
      {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64> }
       ins(%arg3, %arg4: tensor<1x4x16x16xf32>, tensor<16x4x3x3xf32>)
      outs(%out: tensor<1x16x14x14xf32>) -> tensor<1x16x14x14xf32>
    %1 = "tosa.transpose"(%0, %outperms) : (tensor<1x16x14x14xf32>, tensor<4xi32>) -> tensor<1x14x14x16xf32>
    return %1 : tensor<1x14x14x16xf32>
}
