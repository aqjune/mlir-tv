func @f1() -> f32 {
  %c = arith.constant 0xFF800000 : f32
  return %c : f32
}

func @f2() -> f32 {
  %c = arith.constant -0.0 : f32
  return %c : f32
}
