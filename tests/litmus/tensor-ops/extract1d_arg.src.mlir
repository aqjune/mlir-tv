// VERIFY

func.func @extract(%x:f32) -> tensor<2xf32>
{
  %c1 = arith.constant 1.0 : f32
  %c2 = arith.constant 2.0 : f32
  %v = tensor.from_elements %c1, %c2, %x : tensor<3xf32>
  %s = tensor.extract_slice %v[1][2][1]: tensor<3xf32> to tensor<2xf32>
  return %s: tensor<2xf32>
}
