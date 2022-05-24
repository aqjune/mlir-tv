// VERIFY

func.func @f(%arg0: f32, %arg1: f32) -> f32 {
  %i = arith.constant 1.0 : f32
  %v1 = arith.divf %arg0, %i : f32
  %v2 = arith.divf %arg1, %i : f32
  %c = arith.addf %v1, %v2 : f32
  return %c : f32
}
