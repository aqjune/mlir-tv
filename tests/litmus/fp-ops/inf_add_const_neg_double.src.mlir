// VERIFY

func @f() -> f64 {
  %inf = constant 0xFFF0000000000000 : f64
  %v = constant 3.0 : f64
  %v1 = addf %inf, %v : f64
  %v2 = addf %v, %inf : f64
  %c = addf %v1, %v2 : f64
  return %c : f64
}
