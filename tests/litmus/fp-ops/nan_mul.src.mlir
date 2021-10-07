// VERIFY

func @f(%arg0: f32) -> f32 {
  %nan = constant 0x7FC00000 : f32
  %c = mulf %nan, %arg0 : f32
  return %c : f32
}
