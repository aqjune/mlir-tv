// VERIFY

func @f() -> f32 {
  %i = constant 1.0 : f32
  %v = constant 3.0 : f32
  %v1 = mulf %i, %v : f32
  %v2 = mulf %v, %i : f32
  %c = mulf %v1, %v2 : f32
  return %c : f32
}
