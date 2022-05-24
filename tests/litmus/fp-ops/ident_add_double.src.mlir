// VERIFY

func.func @f(%arg0: f64, %arg1: f64) -> f64 {
  %i = arith.constant -0.0 : f64
  %v1 = arith.addf %i, %arg0 : f64
  %v2 = arith.addf %i, %arg1 : f64
  %c = arith.addf %v1, %v2 : f64
  return %c : f64
}
