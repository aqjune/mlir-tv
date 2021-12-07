func @ceil() -> f32 {
  %a = arith.constant 3.00000023842 : f32
  return %a: f32
}

func @floor() -> f32 {
  %a = arith.constant 3.0 : f32
  return %a: f32
}
