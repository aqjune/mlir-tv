func @i32_to_i16() -> i16 {
  %c = arith.constant 0x1234: i16
  return %c: i16
}

func @i64_to_i32() -> i32 {
  %c = arith.constant 0x10001000: i32
  return %c: i32
}

func @i64_to_i16() -> i16 {
  %c = arith.constant 0: i16
  return %c: i16
}
