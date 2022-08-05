module  {
  memref.global "private" constant @__constant_4x8xf32 : memref<4x8xf32> = sparse<[[1, 2], [3, 4]], [1.000000e+00, 2.000000e+00]>
  func @f() -> tensor<4x8xf32> {
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %0 = memref.get_global @__constant_4x8xf32 : memref<4x8xf32>
    %1 = bufferization.to_tensor %0 : memref<4x8xf32>
    return %1 : tensor<4x8xf32>
  }
}

