func.func @f() -> f32 {
  %c = arith.constant 0xFF800000 : f32
  return %c : f32
}
