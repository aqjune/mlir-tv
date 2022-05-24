func.func @pos() -> f32 {
  %c = arith.constant 1.0 : f32
  return %c : f32
}

func.func @neg() -> f32 {
  %c = arith.constant -1.0 : f32
  return %c : f32
}

