// VERIFY

func @f() -> f32 {
  %nan = constant 0x7FC00000 : f32
  %v = constant 3.0 : f32
  %v1 = addf %nan, %v : f32
  %v2 = addf %v, %nan : f32
  %c = addf %v1, %v2 : f32
  return %c : f32
}
