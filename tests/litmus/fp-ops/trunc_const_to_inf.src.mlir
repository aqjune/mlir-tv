// VERIFY

func @f() -> f32 {
  %a = arith.constant 3.4028236e+38 : f64
  %t = arith.truncf %a: f64 to f32
  return %t: f32
}
