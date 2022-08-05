// VERIFY

func @f() -> i32 {
  %v = arith.constant 3: i32
  %w = arith.constant 2: i32
  %x = arith.muli %v, %w: i32
  return %x: i32
}
