// EXPECT: "correct (source is always undefined)"

func.func @shift_right_signed_ub(%v: i32) -> i32 {
  %amnt = arith.constant 32: i32
  %x = arith.shrsi %v, %amnt: i32
  return %x: i32
}
