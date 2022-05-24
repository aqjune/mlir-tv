// VERIFY

memref.global @gv0 : memref<2xf32>

func.func @f(%a: memref<2xf32>, %i: index, %v: f32) -> f32 {
  %gv = memref.get_global @gv0: memref<2xf32>
  memref.store %v, %gv[%i]: memref<2xf32>
  return %v: f32
}
