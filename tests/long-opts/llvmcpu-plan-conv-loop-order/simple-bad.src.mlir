// VERIFY-INCORRECT
// ARGS: --smt-to=200000 --succinct
// Sometimes Z3's model eval hangs, so add --succinct to suppress printing counter examples..

func.func @conv(%filter: memref<3x3x3x32xf32>, %input: memref<1x225x225x3xf32>,
           %output: memref<1x112x112x32xf32>) {
  linalg.conv_2d_nhwc_hwcf
    {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>}
    ins(%input, %filter: memref<1x225x225x3xf32>, memref<3x3x3x32xf32>)
    outs(%output: memref<1x112x112x32xf32>)
  return
}
