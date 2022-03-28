func @maxpool(%arg0: tensor<1x7x7x1280xf32>) -> tensor<1x1x1x1280xf32> {
  %cst = arith.constant -3.40282347E+38 : f32
  %0 = linalg.init_tensor [1, 1, 1, 1280] : tensor<1x1x1x1280xf32>
  %1 = linalg.fill ins(%cst: f32) outs(%0: tensor<1x1x1x1280xf32>) -> tensor<1x1x1x1280xf32> 
  %2 = linalg.init_tensor [7, 7] : tensor<7x7xf32>
  %3 = linalg.pooling_nhwc_max {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%arg0, %2 : tensor<1x7x7x1280xf32>, tensor<7x7xf32>) outs(%1 : tensor<1x1x1x1280xf32>) -> tensor<1x1x1x1280xf32>
  return %3 : tensor<1x1x1x1280xf32>
}
