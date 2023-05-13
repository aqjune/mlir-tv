// VERIFY
// ARGS: --use-neg-zero

func.func @avgpool(%arg0: tensor<1x13x13x1001xf32>) -> tensor<1x1x1x1001xf32> {
  %0 = "tosa.avg_pool2d"(%arg0) {kernel = [13, 13], pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<1x13x13x1001xf32>) -> tensor<1x1x1x1001xf32>
  return %0 : tensor<1x1x1x1001xf32>
}
