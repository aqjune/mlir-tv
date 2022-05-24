func.func @f() -> f32 {
  %inf = arith.constant 0x7F800000 : f32
  %v = arith.constant -3.0 : f32
  %c = arith.mulf %inf, %v : f32
  return %c : f32
}