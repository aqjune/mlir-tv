func @f() -> f32 {
  %c = constant 0xFF800000 : f32
  return %c : f32
}
