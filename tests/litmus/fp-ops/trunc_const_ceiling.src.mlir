// VERIFY

func @ceil() -> f32 {
  %a = arith.constant 3.0000002 : f64
  %t = arith.truncf %a: f64 to f32
  return %t: f32
}

func @ceil_neg() -> f32 {
  %a = arith.constant -3.0000002 : f64
  %t = arith.truncf %a: f64 to f32
  return %t: f32
}

func @ceil_large() -> f32 {
  %a = arith.constant 131072.01 : f64
  %t = arith.truncf %a: f64 to f32
  return %t: f32
}

func @ceil_small() -> f32 {
  %a = arith.constant 9.4039555e-38 : f64
  %t = arith.truncf %a: f64 to f32
  return %t: f32
}
