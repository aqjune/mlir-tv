func.func @conv(%input: tensor<1x225x225x3xf32>, %filter: tensor<3x3x3x32xf32>,
           %output: tensor<1x112x112x32xf32>) -> tensor<1x112x112x32xf32> {
  %0 = bufferization.to_memref %input : memref<1x225x225x3xf32>
  %1 = bufferization.to_memref %filter : memref<3x3x3x32xf32>
  %2 = bufferization.to_memref %output : memref<1x112x112x32xf32>
  %3 = memref.alloc() : memref<1x112x112x32xf32>
  memref.copy %2, %3 : memref<1x112x112x32xf32> to memref<1x112x112x32xf32>
  linalg.conv_2d_nhwc_hwcf
      {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64> }
       ins(%0, %1: memref<1x225x225x3xf32>, memref<3x3x3x32xf32>)
      outs(%3: memref<1x112x112x32xf32>)
  %4 = bufferization.to_tensor %3 : memref<1x112x112x32xf32>
  return %4 : tensor<1x112x112x32xf32>
}
