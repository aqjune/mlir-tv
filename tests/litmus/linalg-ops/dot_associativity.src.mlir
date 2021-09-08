// UNSUPPORTED

// This optimization does not correct in floating point arithmetic but in practice it's correct
// To verify this, we can give "associative" options to mlir-tv.
// Then it checks equality between two input arguments based on multiset theroy.
// `./build/mlir-tv tests/litmus/linalg-ops/dot_associativity.src.mlir tests/litmus/linalg-ops/dot_associativity.tgt.mlir --associative`

func @f() -> tensor<f32> {
  %i = linalg.init_tensor []: tensor<f32>
  %a = constant sparse<[[0], [1], [2], [3], [4]], [-12.0, 3.0, 2.0, 5.0, 4.0]> : tensor<5xf32>
  %b = constant sparse<[[0], [1], [2], [3], [4]], [1.0, 8.0, 5.0, 6.0, 0.0]> : tensor<5xf32>
  %res = linalg.dot ins(%a, %b : tensor<5xf32>,tensor<5xf32>)
    outs(%i: tensor<f32>) -> tensor<f32>
  return %res : tensor<f32>
}
