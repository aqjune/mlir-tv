func @f() {
  %f0 = arith.constant 1.0: f32
  %c1 = arith.constant 1: index
  %c2 = arith.constant 2: index
  %ptr = memref.alloc(): memref<8x64xf32>
  memref.dealloc %ptr: memref<8x64xf32>
  memref.store %f0, %ptr[%c1, %c2]: memref<8x64xf32>
  return
}
