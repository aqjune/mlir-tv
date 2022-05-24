// VERIFY

func.func @fill_view(%arg0: memref<?xf32, offset: ?, strides: [1]>, %arg1: f32) {
  linalg.fill ins(%arg1: f32) outs(%arg0: memref<?xf32, offset: ?, strides: [1]>)
  return
}
