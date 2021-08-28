// VERIFY-INCORRECT

func @f() -> f32 {
  %ptr = memref.alloc(): memref<8x64xf32>
  %f0 = constant 1.0: f32
  %c1 = constant 1: index
  %c2 = constant 2: index
  memref.store %f0, %ptr[%c1, %c2]: memref<8x64xf32>
  %v = memref.load %ptr[%c1, %c2]: memref<8x64xf32>
  return %v: f32
}
