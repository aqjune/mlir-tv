func @f(%idx: index, %idx2: index) -> (f32, f32) {
  %f1 = arith.constant 1.0: f32
  %f2 = arith.constant 2.0: f32
  return %f1, %f2: f32, f32
}
