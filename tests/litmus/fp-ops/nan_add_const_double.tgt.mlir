func @f() -> f64 {
  %v1 = constant 0x7FF7FFFFFFFFFFFF : f64
  %v2 = constant 0x7FF7FFFFFFFFFFFF : f64
  %c = addf %v1, %v2 : f64
  return %c : f64
}
