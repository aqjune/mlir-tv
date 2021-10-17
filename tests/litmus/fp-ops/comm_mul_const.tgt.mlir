func @f() -> f32 {
  %i = arith.constant -2.0 : f32
  %v = arith.constant 3.0 : f32
  %c = arith.mulf %v, %i : f32
  return %c : f32
}
