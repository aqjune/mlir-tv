// VERIFY

func @f() -> f32 {
  %v = arith.constant 3.0 : f32
  %i = arith.constant -1.0 : f32
  %c = arith.divf %v, %i : f32
  return %c : f32
}
