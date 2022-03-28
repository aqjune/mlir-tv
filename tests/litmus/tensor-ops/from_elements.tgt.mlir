func @from_elem() -> tensor<3xf32>
{
  %v = linalg.init_tensor[3]: tensor<3xf32>
  %c0 = arith.constant 3.0: f32
  %res = linalg.fill ins(%c0: f32) outs(%v: tensor<3xf32>) -> tensor<3xf32>
  return %res : tensor<3xf32>
}
