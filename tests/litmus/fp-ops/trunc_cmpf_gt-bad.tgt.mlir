func.func @gt(%arg0: f64, %arg1: f64) -> i1 {
  %e = arith.cmpf "ogt", %arg0, %arg1 : f64
  return %e: i1
}
