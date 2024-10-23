func.func @f(%a: tensor<5xf32>, %b: tensor<5xf32>, %c: tensor<5xf32>, %d: tensor<5xf32>) -> f32 {
  %identity = arith.constant -0.0 : f32
  %i = tensor.empty (): tensor<f32>
  %outty = linalg.fill ins(%identity: f32) outs(%i: tensor<f32>) -> tensor<f32>

  %ca = "tosa.concat"(%a, %c) {axis = 0: i32}: (tensor<5xf32>, tensor<5xf32>) -> tensor<10xf32>
  %cb = "tosa.concat"(%b, %d) {axis = 0: i32}: (tensor<5xf32>, tensor<5xf32>) -> tensor<10xf32>

  %rt = linalg.dot ins(%ca, %cb : tensor<10xf32>, tensor<10xf32>)
      outs(%outty: tensor<f32>) -> tensor<f32>
  %ret = tensor.extract %rt[] : tensor<f32>
  return %ret : f32
}
