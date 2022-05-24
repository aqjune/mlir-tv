module  {
  memref.global "private" constant @__constant_5xf32 : memref<5xf32> = dense<4.200000e+01>
  func.func @f() -> f32 {
    %c0 = arith.constant 0 : index
    %0 = memref.get_global @__constant_5xf32 : memref<5xf32>
    %cst = arith.constant 4.200000e+01 : f32
    return %cst : f32
  }
}

