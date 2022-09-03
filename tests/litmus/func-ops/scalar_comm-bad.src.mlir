// VERIFY-INCORRECT

func.func private @simpl2(%v1: f32, %v2: f32) -> f32

func.func @comm(%v1: f32, %v2: f32) -> f32 {
  %r = func.call @simpl2(%v1, %v2): (f32, f32) -> f32
  return %r: f32
}
