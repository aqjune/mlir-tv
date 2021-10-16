// VERIFY

func @f(%v: i32, %w: i32) -> i32 {
  %x = addi %w, %v: i32
  return %x: i32
}
