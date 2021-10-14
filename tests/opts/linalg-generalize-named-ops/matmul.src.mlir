// VERIFY

func @f(%A : tensor<16x8xi4>, %B: tensor<8x32xi4>, %C: tensor<16x32xi4>) -> tensor<16x32xi4> {
  %0 = linalg.matmul ins(%A, %B: tensor<16x8xi4>, tensor<8x32xi4>)
                          outs(%C: tensor<16x32xi4>) -> tensor<16x32xi4>
  return %0: tensor<16x32xi4>
}
