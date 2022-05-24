memref.global constant @gv0 : memref<2x3xf32> = dense<[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]>

func.func @f() -> f32 {
  %one = arith.constant 1: index
  %two = arith.constant 2: index
  %gv = memref.get_global @gv0: memref<2x3xf32>
  %v = memref.load %gv[%one, %two]: memref<2x3xf32>
  return %v: f32
}
