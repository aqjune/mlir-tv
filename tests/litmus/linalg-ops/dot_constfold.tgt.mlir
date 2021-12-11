// VERIFY

func @f() -> f32 {
  %c0 = arith.constant 1.0: f32
  return %c0 : f32
}
