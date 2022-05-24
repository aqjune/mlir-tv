// VERIFY

func.func @f(%x: f32) -> f32 {
  %y = math.exp %x: f32
  return %y: f32
}
