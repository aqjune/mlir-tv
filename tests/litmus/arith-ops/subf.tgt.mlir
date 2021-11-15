func @f(%v: f32, %w: f32) -> f32 {
  %x = arith.negf %w: f32
  %y = arith.addf %v, %x: f32
  return %y: f32
}
