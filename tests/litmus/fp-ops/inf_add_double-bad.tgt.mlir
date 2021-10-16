func @f(%arg0: f64, %arg1: f64) -> f64 {
  %inf = constant 0x7F800000 : f64
  %c = addf %inf, %inf : f64
  return %c: f64
}
