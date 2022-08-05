// VERIFY-INCORRECT

func @gt(%arg0: f64, %arg1: f64) -> i1 {
  %lhs = arith.truncf %arg0: f64 to f32
  %rhs = arith.truncf %arg1: f64 to f32
  %e = arith.cmpf "ogt", %lhs, %rhs : f32
  return %e: i1
}
