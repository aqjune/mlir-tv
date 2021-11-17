// VERIFY-INCORRECT

func @f() -> f32 {
  %a = arith.constant 3.0 : f64
  %t = arith.truncf %a: f64 to f32
  return %t: f32
}
