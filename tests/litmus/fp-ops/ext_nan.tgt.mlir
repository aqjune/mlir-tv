func.func @f() -> f64 {
  %a = arith.constant 0x7FF7FFFFFFFFFFFF : f64
  return %a: f64
}
