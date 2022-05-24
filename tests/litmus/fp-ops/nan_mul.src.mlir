// VERIFY

func.func @f(%arg0: f32) -> f32 {
  %nan = arith.constant 0x7FC00000 : f32
  %c = arith.mulf %nan, %arg0 : f32
  return %c : f32
}
