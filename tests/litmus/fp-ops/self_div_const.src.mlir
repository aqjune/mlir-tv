// VERIFY

func @pos() -> f32 {
  %v = arith.constant 3.0 : f32
  %c = arith.divf %v, %v : f32
  return %c : f32
}

func @neg() -> f32 {
  %v = arith.constant 3.0 : f32
  %n = arith.constant -3.0 : f32
  %c = arith.divf %v, %n : f32
  return %c : f32
}
