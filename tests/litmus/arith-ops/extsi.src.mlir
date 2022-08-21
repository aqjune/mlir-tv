// VERIFY

func @i32_to_i64() -> i64 {
  %c = arith.constant 2147483647: i32
  %x = arith.extsi %c: i32 to i64
  return %x: i64
}

func @i16_to_i64() -> i64 {
  %c = arith.constant 32767: i16
  %x = arith.extsi %c: i16 to i64
  return %x: i64
}

func @neg_i32_to_i64() -> i64 {
  %c = arith.constant 2147483648: i32
  %x = arith.extsi %c: i32 to i64
  return %x: i64
}
