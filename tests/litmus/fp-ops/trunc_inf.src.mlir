// VERIFY

func @f() -> f32 {
  %inf = arith.constant 0x7FF0000000000000 : f64
  %tinf = arith.truncf %inf: f64 to f32
  return %tinf: f32
}
