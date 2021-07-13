module  {
  func @single_loop_nothing_invariant() {
    %0 = memref.alloc() : memref<10xf32>
    %1 = memref.alloc() : memref<10xf32>
    affine.for %arg0 = 0 to 10 {
      %2 = affine.load %0[%arg0] : memref<10xf32>
      %3 = affine.load %1[%arg0] : memref<10xf32>
      %4 = addf %2, %3 : f32
      affine.store %4, %0[%arg0] : memref<10xf32>
    }
    return
  }
}

