func @f(%x: f32) -> f32 {
  %y = math.abs %x: f32
  return %y: f32
}
