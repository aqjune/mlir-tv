// VERIFY
// ARGS: --use-neg-zero

func.func @maxpool(%arg0: tensor<1x7x7x1280xf32>) -> tensor<1x1x1x1280xf32> {
  %0 = "tosa.max_pool2d"(%arg0) {kernel = array<i64: 7, 7>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<1x7x7x1280xf32>) -> tensor<1x1x1x1280xf32>
  return %0 : tensor<1x1x1x1280xf32>
}
