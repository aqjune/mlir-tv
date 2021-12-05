// VERIFY
// ARGS: --smt-to=20000
func @conv(%input: tensor<1x3x225x225xf32>, %filter: tensor<32x3x3x3xf32>,
           %output: tensor<1x32x112x112xf32>) -> tensor<1x32x112x112xf32> {
  %0 = linalg.conv_2d_nchw_fchw 
      {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64> }
       ins(%input, %filter: tensor<1x3x225x225xf32>, tensor<32x3x3x3xf32>)
      outs(%output: tensor<1x32x112x112xf32>) -> tensor<1x32x112x112xf32>
  return %0 : tensor<1x32x112x112xf32>
}
