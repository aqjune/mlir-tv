func @f() -> f32 {
  %v1 = constant 0x7F800000 : f32
  %v2 = constant 0x7F800000 : f32
  %c = addf %v1, %v2 : f32
  return %c : f32
}
