func @f(%arg0: f32, %arg1: f32) -> f32 {
  %inf = constant 0x7F800000 : f32
  %c = addf %inf, %inf : f32
  return %c: f32
}
