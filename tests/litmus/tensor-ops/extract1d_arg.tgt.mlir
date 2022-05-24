// VERIFY

func.func @extract(%x:f32) -> tensor<2xf32>
{
  %c2 = arith.constant 2.0 : f32
  %v = tensor.from_elements %c2, %x : tensor<2xf32>
  return %v: tensor<2xf32>
}
