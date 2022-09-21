// EXPECT: "correct (source is always undefined)"

func.func @shift_left_index_ub(%v: index) -> index {
  %amnt = arith.constant 32: index
  %x = arith.shli %v, %amnt: index
  return %x: index
}
