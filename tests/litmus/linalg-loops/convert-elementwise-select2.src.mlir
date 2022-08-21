// VERIFY

func @select(%arg0: tensor<8xi1>, %arg1: tensor<8xi32>, %arg2: tensor<8xi32>) -> tensor<8xi32> {
  %0= select %arg0, %arg1, %arg2 : tensor<8xi1>, tensor<8xi32>
  return %0 : tensor<8xi32>
}

// How to reproduce tgt:
// mlir-opt -convert-elementwise-to-linalg <src>
