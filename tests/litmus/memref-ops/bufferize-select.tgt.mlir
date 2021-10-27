module  {
  func @select(%arg0: i1, %arg1: tensor<f32>, %arg2: tensor<f32>) -> tensor<f32> {
    %0 = memref.buffer_cast %arg1 : memref<f32>
    %1 = memref.buffer_cast %arg2 : memref<f32>
    %2 = select %arg0, %0, %1 : memref<f32>
    %3 = memref.tensor_load %2 : memref<f32>
    return %3 : tensor<f32>
  }
}

