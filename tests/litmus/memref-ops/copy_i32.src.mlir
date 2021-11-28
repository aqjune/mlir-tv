// VERIFY

func @copy(%m1: memref<10x10xi32>, %m2: memref<10x10xi32>) -> (tensor<10x10xi32>, tensor<10x10xi32>)
{
  linalg.copy(%m1, %m2) : memref<10x10xi32>, memref<10x10xi32>
  %t1 = bufferization.to_tensor %m1 : memref<10x10xi32>
  %t2 = bufferization.to_tensor %m2 : memref<10x10xi32>
  return %t1, %t2 : tensor<10x10xi32>, tensor<10x10xi32>
}
