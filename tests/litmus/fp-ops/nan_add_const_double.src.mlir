// VERIFY

func @f() -> f64 {
  %nan = constant 0x7FF7FFFFFFFFFFFF : f64
  %v = constant 3.0 : f64
  %v1 = addf %nan, %v : f64
  %v2 = addf %v, %nan : f64
  %c = addf %v1, %v2 : f64
  return %c : f64
}
