// VERIFY-INCORRECT

// This is incorrect if arg0 is NaN or Inf

func.func @f(%arg0: f32) -> f64 {
  %e = arith.extf %arg0: f32 to f64
  %n = arith.negf %arg0 : f32
  %ne = arith.extf %n: f32 to f64
  %s = arith.addf %e, %ne : f64
  return %s: f64
}
