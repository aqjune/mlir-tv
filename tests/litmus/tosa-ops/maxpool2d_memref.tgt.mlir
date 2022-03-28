func @maxpool(%arg0: tensor<1x7x7x1280xf32>) -> tensor<1x1x1x1280xf32> {
  %0 = bufferization.to_memref %arg0 : memref<1x7x7x1280xf32>
  %cst = arith.constant -3.40282347E+38 : f32
  %1 = memref.alloc() : memref<1x1x1x1280xf32>
  linalg.fill ins(%cst: f32) outs(%1: memref<1x1x1x1280xf32>)
  %2 = memref.alloc() : memref<7x7xf32>
  linalg.pooling_nhwc_max {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%0, %2 : memref<1x7x7x1280xf32>, memref<7x7xf32>) outs(%1 : memref<1x1x1x1280xf32>)
  %3 = bufferization.to_tensor %1 : memref<1x1x1x1280xf32>
  return %3 : tensor<1x1x1x1280xf32>
}
