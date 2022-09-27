// VERIFY

func.func @f() -> f32 {
  %a = arith.constant 3.0 : f64
  %t = arith.truncf %a: f64 to f32
  %n = arith.constant -3.0 : f64
  %nt = arith.truncf %n: f64 to f32
  %s = arith.addf %t, %nt : f32
  return %s: f32
}

func.func @tensor() -> tensor<5xf32> {
  %c = arith.constant dense<5.0> : tensor<5xf64>
  %x = arith.truncf %c: tensor<5xf64> to tensor<5xf32>
  return %x: tensor<5xf32>
}
