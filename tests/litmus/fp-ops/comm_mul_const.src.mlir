// VERIFY

func @f() -> f32 {
  %i = arith.constant -2.0 : f32
  %v = arith.constant 3.0 : f32
  %c = arith.mulf %i, %v : f32
  return %c : f32
}
