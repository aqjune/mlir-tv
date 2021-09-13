// VERIFY

func @f() -> f32 {
  %i = constant 0.0 : f32
  %v = constant 3.0 : f32
  %v1 = addf %i, %v : f32
  %v2 = addf %v, %i : f32
  %c = addf %v1, %v2 : f32
  return %c : f32
}
