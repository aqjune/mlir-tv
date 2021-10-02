// VERIFY

func @f() -> f32 {
  %inf = constant 0x7F800000 : f32
  %v = constant 3.0 : f32
  %v1 = addf %inf, %v : f32
  %v2 = addf %v, %inf : f32
  %c = addf %v1, %v2 : f32
  return %c : f32
}
