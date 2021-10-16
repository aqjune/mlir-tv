// VERIFY

func @f(%arg0: f64, %arg1: f64) -> f64 {
  %i = constant -0.0 : f64
  %v1 = addf %i, %arg0 : f64
  %v2 = addf %i, %arg1 : f64
  %c = addf %v1, %v2 : f64
  return %c : f64
}
