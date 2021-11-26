func @conv2d3(%arg0: tensor<2x8x9x3xf32>, %arg1: tensor<?x?x?x?xf32>, %arg2: tensor<5xf32>) {
    %0 = "tosa.conv2d"(%arg0, %arg1, %arg2) {dilation = [1, 1], pad = [0, 0, 0, 0], stride = [1, 1]} : (tensor<2x8x9x3xf32>, tensor<?x?x?x?xf32>, tensor<5xf32>) -> tensor<2x?x?x5xf32>
    return
}