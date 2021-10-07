// VERIFY

func @f() -> f32 {
  %inf = constant 0xFF800000 : f32
  %v = constant 3.0 : f32
  %c = mulf %inf, %v : f32
  return %c : f32
}
