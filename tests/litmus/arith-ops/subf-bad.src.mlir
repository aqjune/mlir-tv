// VERIFY-INCORRECT

func.func @f(%v: f32, %w: f32) -> f32 {
  %x = arith.subf %v, %w: f32
  return %x: f32
}
