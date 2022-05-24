func.func @f(%a: memref<1xf32>) {
  %c0 = arith.constant 0: index
  %v2 = arith.constant 1.2: f32
  memref.store %v2, %a[%c0]: memref<1xf32>
  return
}
