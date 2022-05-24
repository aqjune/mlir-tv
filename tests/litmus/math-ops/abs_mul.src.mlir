// VERIFY

func.func @f(%x: f32, %y: f32) -> f32 {
  %p = math.abs %x: f32
  %q = math.abs %y: f32
  %w = arith.mulf %p, %q: f32
  return %w: f32
}
