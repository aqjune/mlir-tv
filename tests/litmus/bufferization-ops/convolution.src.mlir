// VERIFY

func @conv(%input: tensor<1x225x225x3xf32>, %filter: tensor<3x3x3x32xf32>,
           %output: tensor<1x112x112x32xf32>) -> tensor<1x112x112x32xf32> {
  %0 = linalg.conv_2d_nhwc_hwcf
      {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64> }
       ins(%input, %filter: tensor<1x225x225x3xf32>, tensor<3x3x3x32xf32>)
      outs(%output: tensor<1x112x112x32xf32>) -> tensor<1x112x112x32xf32>
  return %0 : tensor<1x112x112x32xf32>
}
