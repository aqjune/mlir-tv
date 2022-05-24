// VERIFY

func.func @f() -> f32 {
  %ptr = memref.alloca(): memref<8x64xf32>
  %f0 = arith.constant 1.0: f32
  %c1 = arith.constant 1: index
  %c2 = arith.constant 2: index
  memref.store %f0, %ptr[%c1, %c2]: memref<8x64xf32>
  %v = memref.load %ptr[%c1, %c2]: memref<8x64xf32>
  return %v: f32
}
