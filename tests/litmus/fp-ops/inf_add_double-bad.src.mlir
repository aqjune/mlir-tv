// VERIFY-INCORRECT

// This is incorrect if at least one of arg0 and arg1 is NaN or -Inf

func @f(%arg0: f64, %arg1: f64) -> f64 {
  %inf = constant 0x7F800000 : f64
  %v1 = addf %inf, %arg0 : f64
  %v2 = addf %inf, %arg1 : f64
  %c = addf %v1, %v2 : f64
  return %c : f64
}
