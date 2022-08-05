// VERIFY-INCORRECT

func @f() -> f32 {
  %p = arith.constant 3.0 : f32
  %n = arith.negf %p : f32
  return %n : f32
}
