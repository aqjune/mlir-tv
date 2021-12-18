// VERIFY
// ARGS: --smt-to 100000
func @depthwise_conv(%arg0 : tensor<2x7x5x3xf32>, %arg1 : tensor<3x1x3x11xf32>, %arg2 : tensor<33xf32>) -> tensor<2x5x5x11xf32> {
  %0 = "tosa.depthwise_conv2d"(%arg0, %arg1, %arg2) { pad = [0, 0, 0, 0], stride = [1, 1], dilation = [1, 1] } : (tensor<2x7x5x3xf32>, tensor<3x1x3x11xf32>, tensor<33xf32>)  -> (tensor<2x5x5x33xf32>)
  %1 = tensor.extract_slice %0[0,0,0,22][2,5,5,11][1,1,1,1]: tensor<2x5x5x33xf32> to tensor<2x5x5x11xf32>
  return %1 : tensor<2x5x5x11xf32>
}