func @f(%arg0: f64, %arg1: f64) -> f64 {
  %nan = constant 0x7FF7FFFFFFFFFFFF : f64
  %c = addf %nan, %nan : f64
  return %c: f64
}
