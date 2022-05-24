// VERIFY

func.func @f(%arg0: f32, %arg1: f32) -> f32 {
  %i = arith.constant 1.0 : f32
  %v1 = arith.mulf %i, %arg0 : f32
  %v2 = arith.mulf %i, %arg1 : f32
  %c = arith.addf %v1, %v2 : f32
  return %c : f32
}
