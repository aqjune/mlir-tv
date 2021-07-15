module  {
  func @fold_tensor_extract(%arg0: memref<?x?xf32>) -> f32 {
    %c1 = constant 1 : index
    %c2 = constant 2 : index
    %0 = memref.load %arg0[%c1, %c2] : memref<?x?xf32>
    return %0 : f32
  }
}

