func @max_pool2d_is_noop(%arg0: tensor<10x1x1x3xf32>) -> tensor<10x1x1x3xf32> {
  return %arg0 : tensor<10x1x1x3xf32>
}
