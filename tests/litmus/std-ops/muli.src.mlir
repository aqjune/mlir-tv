func @f() -> i32 {
  %v = constant 3: i32
  %w = constant 2: i32
  %x = muli %v, %w: i32
  return %x: i32
}
