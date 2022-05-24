// VERIFY

func.func @f() -> f64 {
  %a = arith.constant 3.0 : f32
  %e = arith.extf %a: f32 to f64
  %n = arith.constant -3.0 : f32
  %ne = arith.extf %n: f32 to f64
  %s = arith.addf %e, %ne : f64
  return %s: f64
}
