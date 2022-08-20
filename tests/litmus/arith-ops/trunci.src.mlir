// VERIFY

func.func @i32_to_i16() -> i16 {
  %c = arith.constant 0xabcd1234: i32
  %x = arith.trunci %c: i32 to i16
  return %x: i16
}

func.func @i64_to_i32() -> i32 {
  %c = arith.constant 0x1234567810001000: i64
  %x = arith.trunci %c: i64 to i32
  return %x: i32
}

func.func @i64_to_i16() -> i16 {
  %c = arith.constant 0x1234567811110000: i64
  %x = arith.trunci %c: i64 to i16
  return %x: i16
}
