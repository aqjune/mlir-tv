func @conv(%arg0: tensor<1x16x16x4xf32>, %arg1: tensor<3x3x4x16xf32>) -> tensor<1x14x14x16xf32> {
    %i = linalg.init_tensor [1,14,14,16] : tensor<1x14x14x16xf32>
    %zero = arith.constant -0.0 : f32
    %out = linalg.fill(%zero, %i) : f32, tensor<1x14x14x16xf32> -> tensor<1x14x14x16xf32>
    %0 = linalg.conv_2d_nhwc_hwcf
      {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64> }
       ins(%arg0, %arg1: tensor<1x16x16x4xf32>, tensor<3x3x4x16xf32>)
      outs(%out: tensor<1x14x14x16xf32>) -> tensor<1x14x14x16xf32>
    return %0 : tensor<1x14x14x16xf32>
}
