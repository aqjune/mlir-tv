// VERIFY

func.func @f() -> i32 {
  %v = arith.constant 3: i32
  %w = arith.constant -1: i32
  %x = arith.xori %v, %w: i32
  return %x: i32
}
