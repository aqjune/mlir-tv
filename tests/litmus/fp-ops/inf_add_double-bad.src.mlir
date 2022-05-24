// VERIFY-INCORRECT

// This is incorrect if at least one of arg0 and arg1 is NaN or -Inf

func.func @f(%arg0: f64, %arg1: f64) -> f64 {
  %inf = arith.constant 0x7F800000 : f64
  %v1 = arith.addf %inf, %arg0 : f64
  %v2 = arith.addf %inf, %arg1 : f64
  %c = arith.addf %v1, %v2 : f64
  return %c : f64
}
