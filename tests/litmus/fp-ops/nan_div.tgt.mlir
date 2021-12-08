func @f1(%arg0: f32) -> f32 {
  %nan = arith.constant 0x7FC00000 : f32
  return %nan: f32
}

func @f2(%arg0: f32) -> f32 {
  %nan = arith.constant 0x7FC00000 : f32
  return %nan: f32
}
