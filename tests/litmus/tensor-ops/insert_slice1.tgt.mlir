func @insert_slice1(%arg0: tensor<1x5x10x15xf32>, %arg1: tensor<1x5x10x15xf32>) -> tensor<1x5x10x15xf32> {
    return %arg0 : tensor<1x5x10x15xf32>
}