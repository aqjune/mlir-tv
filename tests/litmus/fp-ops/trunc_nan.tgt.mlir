func @f() -> f32 {
  %a = arith.constant 0x7FC00000 : f32
  return %a: f32
}
