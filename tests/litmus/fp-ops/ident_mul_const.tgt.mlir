func.func @f() -> f32 {
  %v1 = arith.constant 3.0 : f32
  %v2 = arith.constant 3.0 : f32
  %c = arith.mulf %v1, %v2 : f32
  return %c : f32
}
