// VERIFY

func.func @f(%arg: tensor<3x3xf32>) -> tensor<3x3xf32> {
  %t = tensor.empty (): tensor<3x3xf32>
  %c0 = arith.constant 0.0: f32
  %res = linalg.fill ins(%c0: f32) outs(%t: tensor<3x3xf32>) -> tensor<3x3xf32>
  return %res: tensor<3x3xf32>
}
