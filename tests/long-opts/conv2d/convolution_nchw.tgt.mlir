func @conv(%input: tensor<1x3x225x225xf32>, %filter: tensor<32x3x3x3xf32>,
           %output: tensor<1x32x112x112xf32>) -> tensor<1x32x112x112xf32> {
  %0 = bufferization.to_memref %input : memref<1x3x225x225xf32>
  %1 = bufferization.to_memref %filter : memref<32x3x3x3xf32>
  %2 = bufferization.to_memref %output : memref<1x32x112x112xf32>
  %3 = memref.alloc() : memref<1x32x112x112xf32>
  memref.copy %2, %3 : memref<1x32x112x112xf32> to memref<1x32x112x112xf32>
  linalg.conv_2d_nchw_fchw
      {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64> }
       ins(%0, %1: memref<1x3x225x225xf32>, memref<32x3x3x3xf32>)
      outs(%3: memref<1x32x112x112xf32>)
  %4 = bufferization.to_tensor %3 : memref<1x32x112x112xf32>
  return %4 : tensor<1x32x112x112xf32>
}
