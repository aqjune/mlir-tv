// VERIFY

func @f() -> f64 {
  %inf_p = constant 0x7FF0000000000000 : f64
  %inf_n = constant 0xFFF0000000000000 : f64
  %c = addf %inf_p, %inf_n : f64
  return %c : f64
}
