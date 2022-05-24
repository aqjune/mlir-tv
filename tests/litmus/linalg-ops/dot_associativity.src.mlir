// VERIFY
// ARGS: --associative

// Even if fp addition is not associative in the precise definition of floating
// point arithmetics, people might want to allow rewritings based on it for
// performance.
// To conditionally allow this, we add --associative option to mlir-tv.
// Given the flag, mlir-tv checks the equality between two input arguments of
// dot based on multiset theroy (to be precise, the argument of 'sum').

func.func @f() -> tensor<f32> {
  %i = linalg.init_tensor []: tensor<f32>
  %a = arith.constant sparse<[[0], [1], [2], [3], [4]], [-12.0, 3.0, 2.0, 5.0, 4.0]> : tensor<5xf32>
  %b = arith.constant sparse<[[0], [1], [2], [3], [4]], [1.0, 8.0, 5.0, 6.0, 0.0]> : tensor<5xf32>
  %res = linalg.dot ins(%a, %b : tensor<5xf32>,tensor<5xf32>)
    outs(%i: tensor<f32>) -> tensor<f32>
  return %res : tensor<f32>
}
