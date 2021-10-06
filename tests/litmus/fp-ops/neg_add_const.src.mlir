// VERIFY

func @f() -> f32 {
  %p = constant 3.0 : f32
  %n = constant -3.0 : f32
  %c = addf %p, %n : f32
  return %c : f32
}
