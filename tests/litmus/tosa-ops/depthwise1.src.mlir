// VERIFY

func @conv() -> tensor<1x1x1x1xf32> {
    %img = arith.constant dense<[[[[-0.0],[-0.0],[1.0]],[[-0.0],[-0.0],[-0.0]],[[-0.0],[-0.0],[-0.0]]]]> : tensor<1x3x3x1xf32>
    %filter = arith.constant dense<[[[[1.0]],[[1.0]],[[1.0]]],[[[1.0]],[[1.0]],[[1.0]]],[[[1.0]],[[1.0]],[[1.0]]]]> : tensor<3x3x1x1xf32>
    %c0 = arith.constant -0.0 : f32
    %bias = tensor.from_elements %c0: tensor<1xf32>
    %0 = "tosa.depthwise_conv2d"(%img, %filter, %bias) {dilation = [1, 1], pad = [0, 0, 0, 0], stride = [1, 1]} : (tensor<1x3x3x1xf32>, tensor<3x3x1x1xf32>, tensor<1xf32>) -> tensor<1x1x1x1xf32>
    return %0 : tensor<1x1x1x1xf32>
}