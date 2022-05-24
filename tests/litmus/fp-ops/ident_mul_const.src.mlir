// VERIFY

func.func @f() -> f32 {
  %i = arith.constant 1.0 : f32
  %v = arith.constant 3.0 : f32
  %v1 = arith.mulf %i, %v : f32
  %v2 = arith.mulf %v, %i : f32
  %c = arith.mulf %v1, %v2 : f32
  return %c : f32
}
