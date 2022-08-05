// VERIFY-INCORRECT

// %a may point to a block that isn't @gv0

memref.global @gv0 : memref<2xf32>

func @f(%a: memref<2xf32>) -> memref<2xf32> {
  %gv = memref.get_global @gv0: memref<2xf32>
  return %gv: memref<2xf32>
}
