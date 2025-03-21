func.func @depthwise_conv() -> tensor<1x1x1x2xf32> {
  %arg0 =  arith.constant dense<[[[[1.0,1.0,1.0]]]]>: tensor<1x1x1x3xf32>
  %arg1 =  arith.constant dense<[[[[1.0,2.0],[3.0,4.0],[5.0,6.0]]]]>: tensor<1x1x3x2xf32>
  %arg2 =  arith.constant dense<[7.0,8.0,9.0,10.0,11.0,12.0]>: tensor<6xf32>
  %in = tensor.extract_slice %arg0[0,0,0,2][1,1,1,1][1,1,1,1]: tensor<1x1x1x3xf32> to tensor<1xf32>
  %fil = tensor.extract_slice %arg1[0,0,2,0][1,1,1,2][1,1,1,1]: tensor<1x1x3x2xf32> to tensor<2xf32>
  %in2 = tensor.expand_shape %in [[0,1,2,3]] output_shape [1,1,1,1] : tensor<1xf32> into tensor<1x1x1x1xf32>
  %fil2 = tensor.expand_shape %fil [[0,1,2,3]] output_shape [1,1,1,2] : tensor<2xf32> into tensor<1x1x1x2xf32>
  %bias = tensor.extract_slice %arg2[4][2][1]: tensor<6xf32> to tensor<2xf32>
  %filperms = "tosa.const"() {value = dense<[3, 0, 1, 2]> : tensor<4xi32>} : () -> tensor<4xi32>
  %fil3 = "tosa.transpose"(%fil2, %filperms) : (tensor<1x1x1x2xf32>, tensor<4xi32>) -> tensor<2x1x1x1xf32>
  %0 = "tosa.conv2d"(%in2, %fil3, %bias) { pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>, dilation = array<i64: 1, 1> } : (tensor<1x1x1x1xf32>, tensor<2x1x1x1xf32>, tensor<2xf32>)  -> (tensor<1x1x1x2xf32>)
  return %0 : tensor<1x1x1x2xf32>
}
