// VERIFY

func.func @f() {
  %ptr = memref.alloc(): memref<8x64xf32>
  %ptr0 = memref.subview %ptr[0,0][8,64][1,1]: memref<8x64xf32> to memref<8x64xf32>
  memref.dealloc %ptr0: memref<8x64xf32>
  return
}
