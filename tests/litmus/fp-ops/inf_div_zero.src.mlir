// VERIFY

func.func @f1() -> f32 {
  %inf_p = arith.constant 0xFF800000 : f32
  %zero = arith.constant 0.0 : f32
  %c = arith.divf %inf_p, %zero : f32
  return %c : f32
}

func.func @f2() -> f32 {
  %inf_p = arith.constant 0xFF800000 : f32
  %zero = arith.constant 0.0 : f32
  %c = arith.divf %zero, %inf_p : f32
  return %c : f32
}
