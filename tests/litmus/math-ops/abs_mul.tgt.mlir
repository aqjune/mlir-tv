// VERIFY

func @f(%x: f32, %y: f32) -> f32 {
  %z = arith.mulf %x, %y: f32
  %w = math.abs %z: f32
  return %w: f32
}
