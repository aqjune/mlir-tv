// VERIFY

func @conv2d3(%input: tensor<2x8x9x3xf32>, %weights: tensor<?x?x?x?xf32>, %bias: tensor<5xf32>) -> () {
    %0 = "tosa.conv2d"(%input, %weights, %bias) {pad = [0, 0, 0, 0], stride = [1, 1], dilation = [1, 1]} : (tensor<2x8x9x3xf32>, tensor<?x?x?x?xf32>, tensor<5xf32>)  -> (tensor<?x?x?x?xf32>)
    return
}