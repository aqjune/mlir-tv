// VERIFY

func @f(%arg0: f64, %arg1: f64) -> f64 {
  %nan = constant 0x7FF7FFFFFFFFFFFF : f64
  %v1 = addf %nan, %arg0 : f64
  %v2 = addf %nan, %arg1 : f64
  %c = addf %v1, %v2 : f64
  return %c : f64
}
