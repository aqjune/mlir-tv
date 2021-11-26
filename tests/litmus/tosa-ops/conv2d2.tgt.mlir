func @conv2d2(%arg0: tensor<?x?x?x?xf32>, %arg1: tensor<5x3x6x3xf32>, %arg2: tensor<5xf32>) {
    %0 = "tosa.conv2d"(%arg0, %arg1, %arg2) {dilation = [1, 1], pad = [0, 0, 0, 0], stride = [1, 1]} : (tensor<?x?x?x?xf32>, tensor<5x3x6x3xf32>, tensor<5xf32>) -> tensor<?x?x?x5xf32>
    return
}