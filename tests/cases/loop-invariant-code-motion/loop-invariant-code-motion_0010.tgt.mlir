module  {
  func @parallel_loop_with_invariant() {
    %c0 = constant 0 : index
    %c10 = constant 10 : index
    %c1 = constant 1 : index
    %c7_i32 = constant 7 : i32
    %c8_i32 = constant 8 : i32
    %0 = addi %c7_i32, %c8_i32 : i32
    scf.parallel (%arg0, %arg1) = (%c0, %c0) to (%c10, %c10) step (%c1, %c1) {
      %1 = addi %arg0, %arg1 : index
      scf.yield
    }
    return
  }
}

