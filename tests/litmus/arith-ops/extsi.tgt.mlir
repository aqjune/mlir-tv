func.func @i32_to_i64() -> i64 {
  %c = arith.constant 2147483647: i64
  return %c: i64
}

func.func @i16_to_i64() -> i64 {
  %c = arith.constant 32767: i64
  return %c: i64
}

func.func @neg_i32_to_i64() -> i64 {
  %c = arith.constant 0xffffffff80000000: i64
  return %c: i64
}
