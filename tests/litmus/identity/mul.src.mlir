// VERIFY

func @f(%arg0: f32, %arg1: f32) -> f32 {
  %i = constant 1.0 : f32
  %v1 = mulf %i, %arg0 : f32
  %v2 = mulf %i, %arg1 : f32
  %c = addf %v1, %v2 : f32
  return %c : f32
}
