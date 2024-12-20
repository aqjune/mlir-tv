module  {
  func.func @select(%arg0: i1, %arg1: tensor<f32>, %arg2: tensor<f32>) -> tensor<f32> {
    %0 = bufferization.to_memref %arg1 : tensor<f32> to memref<f32>
    %1 = bufferization.to_memref %arg2 : tensor<f32> to memref<f32>
    %2 = arith.select %arg0, %0, %1 : memref<f32>
    %3 = bufferization.to_tensor %2 : memref<f32> to tensor<f32>
    return %3 : tensor<f32>
  }
}

