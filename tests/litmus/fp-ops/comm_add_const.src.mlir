// VERIFY

func @f() -> f32 {
  %i = constant -2.0 : f32
  %v = constant 3.0 : f32
  %c = addf %i, %v : f32
  return %c : f32
}
