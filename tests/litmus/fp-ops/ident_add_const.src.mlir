// VERIFY

func @f() -> f32 {
  %i = arith.constant -0.0 : f32
  %v = arith.constant 3.0 : f32
  %v1 = arith.addf %i, %v : f32
  %v2 = arith.addf %v, %i : f32
  %c = arith.addf %v1, %v2 : f32
  return %c : f32
}
