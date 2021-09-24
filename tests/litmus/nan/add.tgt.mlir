func @f(%arg0: f32, %arg1: f32) -> f32 {
  %nan = constant 0x7FC00000 : f32
  %c = addf %nan, %nan : f32
  return %c: f32
}
