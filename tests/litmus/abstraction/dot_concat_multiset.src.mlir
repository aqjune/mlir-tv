// VERIFY
// ARGS: --associative --multiset --smt-use-all-logic

// dot (A, B) + dot(C, D) → dot(A::C, B::D)
func @f(%a: tensor<5xf32>, %b: tensor<5xf32>, %c: tensor<5xf32>, %d: tensor<5xf32>) -> f32 {
  %identity = arith.constant -0.0 : f32
  %i = linalg.init_tensor []: tensor<f32>
  %outty = linalg.fill ins(%identity: f32) outs(%i: tensor<f32>) -> tensor<f32>
  %rt1 = linalg.dot ins(%a, %b : tensor<5xf32>, tensor<5xf32>)
      outs(%outty: tensor<f32>) -> tensor<f32>
  %rt2 = linalg.dot ins(%c, %d : tensor<5xf32>, tensor<5xf32>)
      outs(%outty: tensor<f32>) -> tensor<f32>
  %ret1 = tensor.extract %rt1[] : tensor<f32>
  %ret2 = tensor.extract %rt2[] : tensor<f32>

  %ret = arith.addf %ret1, %ret2 : f32
  return %ret : f32
}
