// VERIFY-INCORRECT

// This is incorrect if arg0 is NaN or +-Inf

func @f(%arg0: f32) -> f32 {
  %neg = constant -1.0 : f32
  %arg0_neg = mulf %arg0, %neg : f32
  %c = addf %arg0, %arg0_neg : f32
  return %c : f32
}
