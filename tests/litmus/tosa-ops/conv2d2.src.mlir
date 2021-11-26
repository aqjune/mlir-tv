func @conv2d2(%input: tensor<?x?x?x?xf32>, %weights: tensor<5x3x6x3xf32>, %bias: tensor<5xf32>) -> () {
    %0 = "tosa.conv2d"(%input, %weights, %bias) {pad = [0, 0, 0, 0], stride = [1, 1], dilation = [1, 1]} : (tensor<?x?x?x?xf32>, tensor<5x3x6x3xf32>, tensor<5xf32>)  -> (tensor<?x?x?x?xf32>)
    return
}