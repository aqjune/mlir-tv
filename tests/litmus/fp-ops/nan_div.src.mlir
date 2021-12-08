// VERIFY

func @f1(%arg0: f32) -> f32 {
  %nan = arith.constant 0x7FC00000 : f32
  %c = arith.divf %nan, %arg0 : f32
  return %c : f32
}

func @f2(%arg0: f32) -> f32 {
  %nan = arith.constant 0x7FC00000 : f32
  %c = arith.divf %arg0, %nan : f32
  return %c : f32
}