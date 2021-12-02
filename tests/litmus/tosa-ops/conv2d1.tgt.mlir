func @conv(%arg0: tensor<1x225x225x3xf32>, %arg1: tensor<3x3x3x32xf32>) -> tensor<1x112x112x32xf32> {
    %out = linalg.init_tensor [1,112,112,32] : tensor<1x112x112x32xf32>
    %0 = linalg.conv_2d_nhwc_hwcf
      {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64> }
       ins(%arg0, %arg1: tensor<1x225x225x3xf32>, tensor<3x3x3x32xf32>)
      outs(%out: tensor<1x112x112x32xf32>) -> tensor<1x112x112x32xf32>
    return %0 : tensor<1x112x112x32xf32>
}