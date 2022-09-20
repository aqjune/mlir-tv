// VERIFY

func.func @shift_left_i32(%v: i32) -> i32 {
  %amnt = arith.constant 4: i32
  %x = arith.shli %v, %amnt: i32
  return %x: i32
}

func.func @shift_left_i64(%v: i64) -> i64 {
  %amnt = arith.constant 33: i64
  %x = arith.shli %v, %amnt: i64
  return %x: i64
}

func.func @shift_right_signed_i32() -> i32 {
  %v = arith.constant 0xabcd1234: i32
  %amnt = arith.constant 4: i32
  %x = arith.shrsi %v, %amnt: i32
  return %x: i32
}

func.func @shift_right_signed_i64() -> i64 {
  %v = arith.constant 0xabcd123456781111: i64
  %amnt = arith.constant 36: i64
  %x = arith.shrsi %v, %amnt: i64
  return %x: i64
}

func.func @shift_right_unsigned_i32() -> i32 {
  %v = arith.constant 0xabcd1234: i32
  %amnt = arith.constant 4: i32
  %x = arith.shrui %v, %amnt: i32
  return %x: i32
}

func.func @shift_right_unsigned_i64() -> i64 {
  %v = arith.constant 0xabcd123456781111: i64
  %amnt = arith.constant 36: i64
  %x = arith.shrui %v, %amnt: i64
  return %x: i64
}
