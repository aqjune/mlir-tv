// VERIFY-INCORRECT

func @ne(%arg0: f64, %arg1: f64) -> i1 {
  %lhs = arith.truncf %arg0: f64 to f32
  %rhs = arith.truncf %arg1: f64 to f32
  %e = arith.cmpf "one", %lhs, %rhs : f32
  return %e: i1
}
