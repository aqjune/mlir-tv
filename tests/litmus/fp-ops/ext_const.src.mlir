// VERIFY

func.func @f() -> f64 {
  %a = arith.constant 3.0 : f32
  %e = arith.extf %a: f32 to f64
  %n = arith.constant -3.0 : f32
  %ne = arith.extf %n: f32 to f64
  %s = arith.addf %e, %ne : f64
  return %s: f64
}

func.func @tensor() -> tensor<5xf64> {
  %c = arith.constant dense<5.0> : tensor<5xf32>
  %x = arith.extf %c: tensor<5xf32> to tensor<5xf64>
  return %x: tensor<5xf64>
}
