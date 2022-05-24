// VERIFY

func.func @f() -> f32 {
  %nan = arith.constant 0x7FF7FFFFFFFFFFFF : f64
  %tnan = arith.truncf %nan: f64 to f32
  return %tnan: f32
}
