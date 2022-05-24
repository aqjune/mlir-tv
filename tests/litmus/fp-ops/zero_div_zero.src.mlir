// VERIFY

func.func @f() -> f32 {
  %zero_p = arith.constant 0.0 : f32
  %zero_n = arith.constant -0.0 : f32
  %c = arith.divf %zero_p, %zero_n : f32
  return %c : f32
}
