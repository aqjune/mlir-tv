// VERIFY

func.func @f() -> f32 {
  %inf_p = arith.constant 0xFF800000 : f32
  %inf_n = arith.constant 0.0 : f32
  %c = arith.mulf %inf_p, %inf_n : f32
  return %c : f32
}
