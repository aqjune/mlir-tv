// VERIFY-INCORRECT

func.func private @simpl(%v: f32) -> f32

func.func @chain(%v: f32) -> f32 {
  %r1 = func.call @simpl(%v): (f32) -> f32
  %r2 = func.call @simpl(%r1): (f32) -> f32
  return %r2: f32
}
