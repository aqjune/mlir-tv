memref.global @gv0 : memref<2xf32>

func @f(%a: memref<2xf32>) -> memref<2xf32> {
  return %a: memref<2xf32>
}
