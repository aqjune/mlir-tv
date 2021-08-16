// EXPECT: "Return value mismatch (2/2)"

func @f() -> (f32, f32) {
  %c0 = constant 0.0: f32
  %c1 = constant 1.0: f32
  return %c0, %c1: f32, f32
}
