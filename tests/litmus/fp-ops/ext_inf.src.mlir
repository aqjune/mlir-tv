// VERIFY

func.func @f() -> f64 {
  %inf = arith.constant 0x7F800000 : f32
  %einf = arith.extf %inf: f32 to f64
  return %einf: f64
}
