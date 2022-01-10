
// dot (A, B) + dot(C, D) = dot(A::C, B::D)

func @f(%a: tensor<50xf32>, %b: tensor<50xf32>, %c: tensor<100xf32>, %d: tensor<100xf32>) -> f32 {
  %identity = arith.constant -0.0 : f32
  %i = linalg.init_tensor []: tensor<f32>
  %outty = linalg.fill(%identity, %i) : f32, tensor<f32> -> tensor<f32>

  %ca = "tosa.concat"(%a, %c) {axis = 0: i64}: (tensor<50xf32>, tensor<100xf32>) -> tensor<150xf32>
  %cb = "tosa.concat"(%b, %d) {axis = 0: i64}: (tensor<50xf32>, tensor<100xf32>) -> tensor<150xf32>

  %rt = linalg.dot ins(%ca, %cb : tensor<150xf32>, tensor<150xf32>)
    outs(%outty: tensor<f32>) -> tensor<f32>
  %ret = tensor.extract %rt[] : tensor<f32>
  return %ret : f32
}
