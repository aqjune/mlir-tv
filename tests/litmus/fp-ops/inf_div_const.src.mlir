// VERIFY

func.func @f1() -> f32 {
  %inf = arith.constant 0x7F800000 : f32
  %v = arith.constant 3.0 : f32
  %c = arith.divf %inf, %v : f32
  return %c : f32
}

func.func @f2() -> f32 {
  %inf = arith.constant 0x7F800000 : f32
  %v = arith.constant 3.0 : f32
  %c = arith.divf %v, %inf : f32
  return %c : f32
}
