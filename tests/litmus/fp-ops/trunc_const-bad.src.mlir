// VERIFY-INCORRECT

func.func @f() -> f32 {
  %a = arith.constant 3.0 : f64
  %t = arith.truncf %a: f64 to f32
  %n = arith.constant -5.0 : f64
  %nt = arith.truncf %n: f64 to f32
  %s = arith.addf %t, %nt : f32
  return %s: f32
}
