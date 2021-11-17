func @f() -> f64 {
  %a = arith.constant 0x7FF0000000000000 : f64
  return %a: f64
}
