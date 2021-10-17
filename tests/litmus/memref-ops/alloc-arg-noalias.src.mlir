// VERIFY

func @f(%arg: memref<?xf32>, %idx: index, %idx2: index) -> (f32, f32) {
  %local = memref.alloc(): memref<8xf32>
  %f1 = arith.constant 1.0: f32
  %f2 = arith.constant 2.0: f32
  %c1 = arith.constant 0: index
  %c2 = arith.constant 2: index

  memref.store %f1, %arg[%idx]: memref<?xf32>
  memref.store %f2, %local[%idx2]: memref<8xf32>
  %v1 = memref.load %arg[%idx]: memref<?xf32>
  %v2 = memref.load %local[%idx2]: memref<8xf32>
  return %v1, %v2: f32, f32
}
