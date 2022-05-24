func.func @f() -> f64 {
  %v1 = arith.constant 0xFFF0000000000000 : f64
  %v2 = arith.constant 0xFFF0000000000000 : f64
  %c = arith.addf %v1, %v2 : f64
  return %c : f64
}
