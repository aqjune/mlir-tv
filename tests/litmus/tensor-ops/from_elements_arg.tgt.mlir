func.func @from_elem(%x:f32) -> tensor<3xf32>
{
  %v = linalg.init_tensor[3]: tensor<3xf32>
  %res = linalg.fill ins(%x: f32) outs(%v: tensor<3xf32>) -> tensor<3xf32>
  return %res : tensor<3xf32>
}
