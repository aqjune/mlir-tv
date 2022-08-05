func @floor() -> f32 {
  %a = arith.constant 3.0 : f32
  return %a: f32
}

func @floor_neg() -> f32 {
  %a = arith.constant -3.0 : f32
  return %a: f32
}

func @floor_large() -> f32 {
  %a = arith.constant 131072.0 : f32
  return %a: f32
}

func @floor_small() -> f32 {
  %a = arith.constant 9.40395480658e-38 : f32
  return %a: f32
}
