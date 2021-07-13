module  {
  func @variant_loop_dialect() {
    %c0 = constant 0 : index
    %c10 = constant 10 : index
    %c1 = constant 1 : index
    %0 = memref.alloc() : memref<10xf32>
    scf.for %arg0 = %c0 to %c10 step %c1 {
      scf.for %arg1 = %c0 to %c10 step %c1 {
        %1 = addi %arg0, %arg1 : index
      }
    }
    return
  }
}

