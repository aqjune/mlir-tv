func @f() -> f64 {
  %c = constant 0x7FF7FFFFFFFFFFFF : f64
  return %c : f64
}
