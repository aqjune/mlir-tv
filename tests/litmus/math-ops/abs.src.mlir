// VERIFY

func.func @f(%x: f32) -> f32 {
  %y = math.abs %x: f32
  %z = math.abs %y: f32
  return %z: f32
}
