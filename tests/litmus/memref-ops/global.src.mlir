// VERIFY

memref.global @gv0 : memref<2xf32>

func.func @f(%a: memref<2xf32>, %i: index, %v: f32) -> f32 {
  %gv = memref.get_global @gv0: memref<2xf32>
  memref.store %v, %gv[%i]: memref<2xf32>
  %v2 = memref.load %gv[%i]: memref<2xf32>
  return %v2: f32
}
