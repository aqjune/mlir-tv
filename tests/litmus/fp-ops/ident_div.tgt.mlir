func @f(%arg0: f32, %arg1: f32) -> f32 {
  %c = arith.addf %arg0, %arg1 : f32
  return %c: f32
}
