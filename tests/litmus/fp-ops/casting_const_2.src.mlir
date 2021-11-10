// VERIFY

func @f() -> f64 {
  %a = arith.constant 3.0 : f32
  %e = arith.extf %a: f32 to f64
  %d = arith.constant -5.0 : f64
  %ne = arith.constant -3.0 : f64
  %s = arith.addf %e, %ne : f64
  return %s: f64
}

