// VERIFY

func @select(%arg0: tensor<i1>, %arg1: tensor<i32>, %arg2: tensor<i32>) -> tensor<i32> {
  %0 = select %arg0, %arg1, %arg2 : tensor<i1>, tensor<i32>
  return %0 : tensor<i32>
}

// How to reproduce tgt:
// mlir-opt -convert-elementwise-to-linalg <src>
