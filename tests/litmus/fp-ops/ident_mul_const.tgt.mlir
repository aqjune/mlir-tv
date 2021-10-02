func @f() -> f32 {
  %v1 = constant 3.0 : f32
  %v2 = constant 3.0 : f32
  %c = mulf %v1, %v2 : f32
  return %c : f32
}
