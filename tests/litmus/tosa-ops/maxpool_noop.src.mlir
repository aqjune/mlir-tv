// VERIFY

func.func @max_pool2d_is_noop(%arg0: tensor<10x1x1x3xf32>) -> tensor<10x1x1x3xf32> {
  %0 = "tosa.max_pool2d"(%arg0) {kernel = [1, 1], pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>, dilation = array<i64: 1, 1>} : (tensor<10x1x1x3xf32>) -> tensor<10x1x1x3xf32>
  return %0 : tensor<10x1x1x3xf32>
}
