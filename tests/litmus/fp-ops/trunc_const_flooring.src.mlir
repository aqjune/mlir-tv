// VERIFY

func.func @floor() -> f32 {
  %a = arith.constant 3.0000001 : f64
  %t = arith.truncf %a: f64 to f32
  return %t: f32
}

func.func @floor_neg() -> f32 {
  %a = arith.constant -3.0000001 : f64
  %t = arith.truncf %a: f64 to f32
  return %t: f32
}

func.func @floor_large() -> f32 {
  %a = arith.constant 131072.005 : f64
  %t = arith.truncf %a: f64 to f32
  return %t: f32
}

func.func @floor_small() -> f32 {
  %a = arith.constant 9.403955e-38 : f64
  %t = arith.truncf %a: f64 to f32
  return %t: f32
}
