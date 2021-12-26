func @f(%A : tensor<8x4xf32>, %B: tensor<4x16xf32>, %C: tensor<8x16xf32>, %D: tensor<8x16xf32>) -> tensor<8x16xf32> {
  %0 = linalg.matmul ins(%A, %B: tensor<8x4xf32>, tensor<4x16xf32>)
                          outs(%D: tensor<8x16xf32>) -> tensor<8x16xf32>
  return %0: tensor<8x16xf32>
}
