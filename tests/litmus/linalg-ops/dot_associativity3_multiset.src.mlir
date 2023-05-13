// VERIFY
// ARGS: --associative --multiset -fp-bits=5

// Even if fp addition is not associative in the precise definition of floating
// point arithmetics, people might want to allow rewritings based on it for
// performance.
// To conditionally allow this, we add --associative option to mlir-tv.
// Given the flag, mlir-tv checks the equality between two input arguments of
// dot based on multiset theroy (to be precise, the argument of 'sum').

func.func @f() -> (f32, f32) {
  %c0 = arith.constant 0 : index
  %i = tensor.empty (): tensor<f32>
  %a1 = arith.constant sparse<[[0], [1]], [-12.0, 3.0]> : tensor<2xf32>
  %a2 = arith.constant sparse<[[0], [1], [2]], [2.0, 5.0, 4.0]> : tensor<3xf32>
  %b1 = arith.constant sparse<[[0], [1]], [1.0, 8.0]> : tensor<2xf32>
  %b2 = arith.constant sparse<[[0], [1], [2]], [5.0, 6.0, 2.0]> : tensor<3xf32>
  %o1 = linalg.dot ins(%a1, %b1 : tensor<2xf32>,tensor<2xf32>)
    outs(%i: tensor<f32>) -> tensor<f32>
  %o2 = linalg.dot ins(%a2, %b2 : tensor<3xf32>,tensor<3xf32>)
    outs(%i: tensor<f32>) -> tensor<f32>
  %res1 = tensor.extract %o1[] : tensor<f32>
  %res2 = tensor.extract %o2[] : tensor<f32>
  return %res1, %res2 : f32, f32
}
