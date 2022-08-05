
func @transpose() -> tensor<1x1x2x2xf32> {
  %cst = arith.constant dense<[[[[1.0, 2.0], [3.0, 4.0]]]]> : tensor<1x1x2x2xf32>
  return %cst : tensor<1x1x2x2xf32>
}
