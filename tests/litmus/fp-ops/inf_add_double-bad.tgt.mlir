func @f(%arg0: f64, %arg1: f64) -> f64 {
  %inf = arith.constant 0x7F800000 : f64
  %c = arith.addf %inf, %inf : f64
  return %c: f64
}
