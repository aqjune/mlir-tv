// VERIFY

func.func @cmpf(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<i1> {
  %0 = arith.cmpf olt, %arg0, %arg1 : tensor<f32>
  return %0 : tensor<i1>
}

// How to reproduce tgt:
// mlir-opt -convert-elementwise-to-linalg <src>
