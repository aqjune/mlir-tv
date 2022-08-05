// VERIFY

func @fill_view(%arg0: memref<?xf32, offset: ?, strides: [1]>, %arg1: f32) {
  linalg.fill(%arg1, %arg0): f32, memref<?xf32, offset: ?, strides: [1]>
  return
}
