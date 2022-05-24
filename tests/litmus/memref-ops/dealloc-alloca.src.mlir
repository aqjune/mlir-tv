// EXPECT: "correct (source is always undefined)"

func.func @f() {
  %ptr = memref.alloca(): memref<8x64xf32>
  memref.dealloc %ptr: memref<8x64xf32>
  return
}
