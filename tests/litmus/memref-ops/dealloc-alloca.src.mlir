// EXPECT: "correct (source is always undefined)"

func @f() {
  %ptr = memref.alloca(): memref<8x64xf32>
  memref.dealloc %ptr: memref<8x64xf32>
  return
}
