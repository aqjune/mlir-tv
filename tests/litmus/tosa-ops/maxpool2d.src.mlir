// VERIFY
// ARGS: --use-neg-zero

func @maxpool(%arg0: tensor<1x7x7x1280xf32>) -> tensor<1x1x1x1280xf32> {
  %0 = "tosa.max_pool2d"(%arg0) {kernel = [7, 7], pad = [0, 0, 0, 0], stride = [1, 1]} : (tensor<1x7x7x1280xf32>) -> tensor<1x1x1x1280xf32>
  return %0 : tensor<1x1x1x1280xf32>
}
