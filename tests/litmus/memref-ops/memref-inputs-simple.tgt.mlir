func.func @f(%a: memref<i32>, %b: memref<i32>) {
  %c0 = arith.constant 0: i32
  %c1 = arith.constant 1: i32
  memref.store %c1, %b[]: memref<i32>
  memref.store %c0, %a[]: memref<i32>
  return
}
