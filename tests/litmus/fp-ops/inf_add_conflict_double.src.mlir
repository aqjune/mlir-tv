// VERIFY

func @f() -> f64 {
  %inf_p = arith.constant 0x7FF0000000000000 : f64
  %inf_n = arith.constant 0xFFF0000000000000 : f64
  %c = arith.addf %inf_p, %inf_n : f64
  return %c : f64
}
