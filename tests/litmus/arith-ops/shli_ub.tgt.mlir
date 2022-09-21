func.func @shift_left_ub(%v: i32) -> i32 {
  %x = arith.constant 0xffffffff: i32
  return %x: i32
}
