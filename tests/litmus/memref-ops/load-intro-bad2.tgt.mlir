func @f(%a: memref<1xf32>) {
  %c0 = arith.constant 0: index
  // This is UB when %a[0] is uninitialized.
  %dummy = memref.load %a[%c0]: memref<1xf32>
  %v = arith.constant 1.1: f32
  memref.store %v, %a[%c0]: memref<1xf32>
  return
}
