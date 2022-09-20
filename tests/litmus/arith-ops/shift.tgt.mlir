func.func @shift_left_i32(%v: i32) -> i32 {
  %amnt = arith.constant 16: i32
  %x = arith.muli %v, %amnt: i32
  return %x: i32
}

func.func @shift_left_i64(%v: i64) -> i64 {
  %amnt = arith.constant 8589934592: i64
  %x = arith.muli %v, %amnt: i64
  return %x: i64
}

func.func @shift_right_signed_i32() -> i32 {
  %x = arith.constant 0xfabcd123: i32
  return %x: i32
}

func.func @shift_right_signed_i64() -> i64 {
  %x = arith.constant 0xfffffffffabcd123: i64
  return %x: i64
}

func.func @shift_right_unsigned_i32() -> i32 {
  %x = arith.constant 0x0abcd123: i32
  return %x: i32
}

func.func @shift_right_unsigned_i64() -> i64 {
  %x = arith.constant 0x000000000abcd123: i64
  return %x: i64
}
