func.func @ceil() -> f32 {
  %a = arith.constant 3.00000023842 : f32
  return %a: f32
}

func.func @ceil_neg() -> f32 {
  %a = arith.constant -3.00000023842 : f32
  return %a: f32
}

func.func @ceil_large() -> f32 {
  %a = arith.constant 131072.015625 : f32
  return %a: f32
}

func.func @ceil_small() -> f32 {
  %a = arith.constant 9.40395592762e-38 : f32
  return %a: f32
}