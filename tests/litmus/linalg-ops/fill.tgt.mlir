func @f(%arg: tensor<3x3xf32>) -> tensor<3x3xf32> {
  %c0 = constant 0.0: f32
  %res = linalg.fill (%c0, %arg): f32, tensor<3x3xf32> -> tensor<3x3xf32>
  return %res: tensor<3x3xf32>
}
