// VERIFY

func.func @f() -> f32 {
  %z = arith.constant 0.0 : f32
  %v = arith.constant 3.0 : f32
  %c = arith.divf %z, %v : f32
  return %c : f32
}
