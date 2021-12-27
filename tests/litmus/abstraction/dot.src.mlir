// EXPECT: "dot ops (fp): SUM_MUL"

func @f(%a: tensor<?xf32>, %b: tensor<?xf32>) -> tensor<f32> {
  %zero = arith.constant -0.0 : f32
  %i = linalg.init_tensor []: tensor<f32>
  %outty = linalg.fill(%zero, %i) : f32, tensor<f32> -> tensor<f32>
  %e = linalg.dot ins(%a, %b : tensor<?xf32>,tensor<?xf32>)
    outs(%outty: tensor<f32>) -> tensor<f32>
  return %e : tensor<f32>
}
