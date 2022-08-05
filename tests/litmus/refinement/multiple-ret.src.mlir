// EXPECT: "Return value mismatch (2/2)"

func @f() -> (f32, f32) {
  %c0 = arith.constant 0.0: f32
  %c1 = arith.constant 1.0: f32
  return %c0, %c1: f32, f32
}
