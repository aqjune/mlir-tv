// EXPECT: "correct (source is always undefined)"

func.func @shift_left_ub(%v: i32) -> i32 {
  %amnt = arith.constant 32: i32
  %x = arith.shli %v, %amnt: i32
  return %x: i32
}
