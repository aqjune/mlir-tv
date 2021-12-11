// VERIFY

func @ceil() -> f32 {
  %a = arith.constant 3.0000002 : f64
  %t = arith.truncf %a: f64 to f32
  return %t: f32
}

func @floor() -> f32 {
  %a = arith.constant 3.0000001 : f64
  %t = arith.truncf %a: f64 to f32
  return %t: f32
}
