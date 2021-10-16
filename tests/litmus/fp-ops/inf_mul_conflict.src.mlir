// VERIFY

func @f() -> f32 {
  %inf_p = constant 0x7F800000 : f32
  %inf_n = constant 0xFF800000 : f32
  %c = mulf %inf_p, %inf_n : f32
  return %c : f32
}
