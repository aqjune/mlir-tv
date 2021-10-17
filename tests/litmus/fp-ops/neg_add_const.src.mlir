// VERIFY

func @f() -> f32 {
  %p = arith.constant 3.0 : f32
  %n = arith.constant -3.0 : f32
  %c = arith.addf %p, %n : f32
  return %c : f32
}
