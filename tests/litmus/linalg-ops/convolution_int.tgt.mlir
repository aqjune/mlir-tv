func.func @conv() -> tensor<1x1x1x1xf32> {
    %img = arith.constant dense<[[[[1.0,-0.0],[-0.0,-0.0],[-0.0,-0.0]],[[-0.0,-0.0],[-0.0,-0.0],[-0.0,-0.0]],[[-0.0,-0.0],[-0.0,-0.0],[-0.0,-0.0]]]]> : tensor<1x3x3x2xf32>
    %fil = arith.constant dense<[[[[1.0],[1.0]],[[1.0],[1.0]],[[1.0],[1.0]]],[[[1.0],[1.0]],[[1.0],[1.0]],[[1.0],[1.0]]],[[[1.0],[1.0]],[[1.0],[1.0]],[[1.0],[1.0]]]]> : tensor<3x3x2x1xf32>
    %out = tensor.empty () : tensor<1x1x1x1xf32>
    %0 = linalg.conv_2d_nhwc_hwcf
      {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64> }
       ins(%img, %fil: tensor<1x3x3x2xf32>, tensor<3x3x2x1xf32>)
      outs(%out: tensor<1x1x1x1xf32>) -> tensor<1x1x1x1xf32>
    return %0 : tensor<1x1x1x1xf32>
}

