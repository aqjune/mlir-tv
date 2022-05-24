// VERIFY-INCORRECT

func.func @f() {
  %ptr = memref.alloc(): memref<8x64xf32>
  memref.dealloc %ptr: memref<8x64xf32>
  return
}
