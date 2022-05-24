// VERIFY

func.func @f() -> f32 {
  %inf = arith.constant 0x7F800000 : f32
  %v = arith.constant 3.0 : f32
  %v1 = arith.addf %inf, %v : f32
  %v2 = arith.addf %v, %inf : f32
  %c = arith.addf %v1, %v2 : f32
  return %c : f32
}
