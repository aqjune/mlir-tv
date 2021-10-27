// VERIFY

func @select(%arg0: i1, %arg1: tensor<f32>, %arg2: tensor<f32>) -> tensor<f32> {
  %0 = select %arg0, %arg1, %arg2 : tensor<f32>
  return %0 : tensor<f32>
}

// How to reproduce tgt:
// mlir-opt -std-bufferize <src>