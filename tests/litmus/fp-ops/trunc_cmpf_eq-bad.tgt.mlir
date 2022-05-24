func.func @eq(%arg0: f64, %arg1: f64) -> i1 {
  %e = arith.cmpf "oeq", %arg0, %arg1 : f64
  return %e: i1
}
