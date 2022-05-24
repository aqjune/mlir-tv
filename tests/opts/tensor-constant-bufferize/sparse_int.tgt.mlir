module  {
  memref.global "private" constant @__constant_5x1xi32 : memref<5x1xi32> = sparse<[[0, 0], [1, 0], [2, 0]], [-1, -2, -3]>
  func.func @f() -> i32 {
    %c2 = arith.constant 2 : index
    %c0 = arith.constant 0 : index
    %0 = memref.get_global @__constant_5x1xi32 : memref<5x1xi32>
    %c-3_i32 = arith.constant -3 : i32
    return %c-3_i32 : i32
  }
}

