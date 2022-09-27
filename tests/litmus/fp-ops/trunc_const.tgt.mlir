func.func @f() -> f32 {
  %a = arith.constant 0.0 : f32
  return %a: f32
}

func.func @tensor() -> tensor<5xf32> {
  %x = arith.constant dense<5.0> : tensor<5xf32>
  return %x: tensor<5xf32>
}
