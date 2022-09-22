func.func @shift_left_index_ub(%v: index) -> index {
  %x = arith.constant 0xf0f0f0f0: index
  return %x: index
}
