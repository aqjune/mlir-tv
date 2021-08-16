func @f() -> (f32, f32) {
  %c0 = constant 0.0: f32
  %c2 = constant 2.0: f32
  return %c0, %c2: f32, f32
}
