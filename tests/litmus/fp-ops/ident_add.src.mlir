// VERIFY

func @f(%arg0: f32, %arg1: f32) -> f32 {
  %i = arith.constant -0.0 : f32
  %v1 = arith.addf %i, %arg0 : f32
  %v2 = arith.addf %i, %arg1 : f32
  %c = arith.addf %v1, %v2 : f32
  return %c : f32
}
