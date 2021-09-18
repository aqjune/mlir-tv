func @from_elem() -> tensor<3xf32>
{
  %v = linalg.init_tensor[3]: tensor<3xf32>
  %c0 = constant 3.0: f32
  %res = linalg.fill (%c0, %v): f32, tensor<3xf32> -> tensor<3xf32>
  return %res : tensor<3xf32>
}
