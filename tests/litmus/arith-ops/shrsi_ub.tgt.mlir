func.func @shift_right_signed_ub(%v: i32) -> i32 {
  %x = arith.constant 0xf0f0f0f0: i32
  return %x: i32
}
