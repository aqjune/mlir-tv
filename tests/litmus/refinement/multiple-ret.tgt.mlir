func @f() -> (f32, f32) {
  %c0 = arith.constant 0.0: f32
  %c2 = arith.constant 2.0: f32
  return %c0, %c2: f32, f32
}
