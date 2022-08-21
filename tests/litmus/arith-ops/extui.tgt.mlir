func @i32_to_i64() -> i64 {
  %c = arith.constant 2147483647: i64
  return %c: i64
}

func @i16_to_i64() -> i64 {
  %c = arith.constant 32767: i64
  return %c: i64
}

func @neg_i32_to_i64() -> i64 {
  %c = arith.constant 0x0000000080000000: i64
  return %c: i64
}
