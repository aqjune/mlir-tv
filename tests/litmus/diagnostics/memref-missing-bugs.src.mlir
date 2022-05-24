// EXPECT: "mlir-tv assumes that memrefs of different element types do not alias. This can cause missing bugs."

func.func @f(%a: memref<f32>, %b: memref<i32>) {
  return
}
