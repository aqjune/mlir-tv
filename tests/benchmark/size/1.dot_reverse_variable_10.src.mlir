
// dot (reverse(variable1), reverse(variable2)) = dot(variable1, variable2)

func @f(%a: tensor<5xf32>, %b: tensor<5xf32>) -> tensor<f32> {
  %identity = arith.constant -0.0 : f32
  %i = linalg.init_tensor []: tensor<f32>
  %outty = linalg.fill(%identity, %i) : f32, tensor<f32> -> tensor<f32>

  %ra = "tosa.reverse"(%a) {axis = 0 : i64} : (tensor<5xf32>) -> tensor<5xf32>
  %rb = "tosa.reverse"(%b) {axis = 0 : i64} : (tensor<5xf32>) -> tensor<5xf32>
  %res = linalg.dot ins(%ra, %rb : tensor<5xf32>, tensor<5xf32>)
    outs(%outty: tensor<f32>) -> tensor<f32>
  return %res : tensor<f32>
}
