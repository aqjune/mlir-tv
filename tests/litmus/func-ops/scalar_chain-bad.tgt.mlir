func.func private @simpl(%v: f32) -> f32

func.func @chain(%v: f32) -> f32 {
  %r1 = func.call @simpl(%v): (f32) -> f32
  return %r1: f32
}
