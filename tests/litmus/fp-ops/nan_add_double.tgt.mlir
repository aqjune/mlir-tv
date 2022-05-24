func.func @f(%arg0: f64, %arg1: f64) -> f64 {
  %nan = arith.constant 0x7FF7FFFFFFFFFFFF : f64
  %c = arith.addf %nan, %nan : f64
  return %c: f64
}
