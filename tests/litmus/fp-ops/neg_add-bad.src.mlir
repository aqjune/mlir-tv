// VERIFY-INCORRECT

// This is incorrect if arg0 is NaN or +-Inf

func.func @f(%arg0: f32) -> f32 {
  %neg = arith.constant -1.0 : f32
  %arg0_neg = arith.mulf %arg0, %neg : f32
  %c = arith.addf %arg0, %arg0_neg : f32
  return %c : f32
}
