// VERIFY

func @f(%arg0: f32, %arg1: f32) -> f32 {
  %nan = arith.constant 0x7FC00000 : f32
  %v1 = arith.addf %nan, %arg0 : f32
  %v2 = arith.addf %nan, %arg1 : f32
  %c = arith.addf %v1, %v2 : f32
  return %c : f32
}
