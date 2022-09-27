func.func @f() -> f64 {
  %a = arith.constant 0.0 : f64
  return %a: f64
}

func.func @tensor() -> tensor<5xf64> {
  %x = arith.constant dense<5.0> : tensor<5xf64>
  return %x: tensor<5xf64>
}
