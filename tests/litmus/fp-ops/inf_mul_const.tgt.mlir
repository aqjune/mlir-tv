func @f() -> f32 {
  %c = arith.constant 0x7F800000 : f32
  return %c : f32
}
