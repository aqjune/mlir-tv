func.func @conv(%filter: memref<3x3x1x1xf32>,
           %output: memref<1x1x1x1xf32>) {
  %input = memref.alloc(): memref<1x3x3x1xf32>
  linalg.conv_2d_nhwc_hwcf
    {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
    ins(%input, %filter: memref<1x3x3x1xf32>, memref<3x3x1x1xf32>)
    outs(%output: memref<1x1x1x1xf32>)
  return
}
