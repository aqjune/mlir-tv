// VERIFY

func.func @f() -> f64 {
  %inf = arith.constant 0xFFF0000000000000 : f64
  %v = arith.constant 3.0 : f64
  %v1 = arith.addf %inf, %v : f64
  %v2 = arith.addf %v, %inf : f64
  %c = arith.addf %v1, %v2 : f64
  return %c : f64
}
