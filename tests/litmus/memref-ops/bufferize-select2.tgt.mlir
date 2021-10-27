module  {
  func @select(%arg0: i1, %arg1: tensor<?xf32>, %arg2: tensor<?xf32>) -> tensor<?xf32> {
    %0 = memref.buffer_cast %arg1 : memref<?xf32>
    %1 = memref.buffer_cast %arg2 : memref<?xf32>
    %2 = select %arg0, %0, %1 : memref<?xf32>
    %3 = memref.tensor_load %2 : memref<?xf32>
    return %3 : tensor<?xf32>
  }
}

