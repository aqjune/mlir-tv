func @f(%v: f32, %w: f32) -> f32 {
  %x = arith.negf %v: f32
  %y = arith.addf %w, %x: f32
  return %y: f32
}
