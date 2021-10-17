// VERIFY

func @f() -> f64 {
  %nan = arith.constant 0x7FF7FFFFFFFFFFFF : f64
  %v = arith.constant 3.0 : f64
  %v1 = arith.addf %nan, %v : f64
  %v2 = arith.addf %v, %nan : f64
  %c = arith.addf %v1, %v2 : f64
  return %c : f64
}
