// VERIFY

func @f(%arg0: f64, %arg1: f64) -> f64 {
  %nan = arith.constant 0x7FF7FFFFFFFFFFFF : f64
  %v1 = arith.addf %nan, %arg0 : f64
  %v2 = arith.addf %nan, %arg1 : f64
  %c = arith.addf %v1, %v2 : f64
  return %c : f64
}
