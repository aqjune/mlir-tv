module  {
  func.func @select(%arg0: i1, %arg1: tensor<?xf32>, %arg2: tensor<?xf32>) -> tensor<?xf32> {
    %0 = bufferization.to_memref %arg1 : memref<?xf32>
    %1 = bufferization.to_memref %arg2 : memref<?xf32>
    %2 = arith.select %arg0, %0, %1 : memref<?xf32>
    %3 = bufferization.to_tensor %2 : memref<?xf32>
    return %3 : tensor<?xf32>
  }
}

