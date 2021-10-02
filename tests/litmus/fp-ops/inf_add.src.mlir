// VERIFY-INCORRECT

// This may be incorrect if at least one of arg0 and arg1 is NaN or -Inf

func @f(%arg0: f32, %arg1: f32) -> f32 {
  %inf = constant 0x7F800000 : f32
  %v1 = addf %inf, %arg0 : f32
  %v2 = addf %inf, %arg1 : f32
  %c = addf %v1, %v2 : f32
  return %c : f32
}
