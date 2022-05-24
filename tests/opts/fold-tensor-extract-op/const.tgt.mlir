module  {
  func.func @fold_tensor_extract(%arg0: memref<2x3xf32>) -> f32 {
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %0 = memref.load %arg0[%c1, %c2] : memref<2x3xf32>
    return %0 : f32
  }
}

