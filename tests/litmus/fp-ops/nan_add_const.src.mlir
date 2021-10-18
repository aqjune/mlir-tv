// VERIFY

func @f() -> f32 {
  %nan = arith.constant 0x7FC00000 : f32
  %v = arith.constant 3.0 : f32
  %v1 = arith.addf %nan, %v : f32
  %v2 = arith.addf %v, %nan : f32
  %c = arith.addf %v1, %v2 : f32
  return %c : f32
}
