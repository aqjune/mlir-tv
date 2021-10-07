func @f(%arg0: f32) -> f32 {
  %nan = constant 0x7FC00000 : f32
  return %nan: f32
}
