// VERIFY

func @f(%arg0: f64, %arg1: f64) -> f64 {
  %c = addf %arg0, %arg1 : f64
  return %c : f64
}