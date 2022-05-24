func.func @f(%arg0: f64, %arg1: f64) -> f64 {
  %c = arith.addf %arg1, %arg0 : f64
  return %c: f64
}
