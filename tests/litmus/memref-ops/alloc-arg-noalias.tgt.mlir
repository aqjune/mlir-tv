func @f(%arg: memref<?xf32>, %idx: index, %idx2: index) -> (f32, f32) {
  %f1 = constant 1.0: f32
  %f2 = constant 2.0: f32
  %c1 = constant 0: index
  %c2 = constant 2: index

  memref.store %f1, %arg[%idx]: memref<?xf32>
  return %f1, %f2: f32, f32
}
