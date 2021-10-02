func @f() -> f32 {
  %c = constant 0x7FC00000 : f32
  return %c : f32
}
