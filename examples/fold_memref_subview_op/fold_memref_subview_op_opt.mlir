module  {
  func @fold_subview(%arg0: tensor<64x64xf32>, %arg1: index) -> f32 {
    %c0 = constant 0 : index
    %0 = memref.buffer_cast %arg0 : memref<64x64xf32>
    %1 = memref.load %0[%c0, %arg1] : memref<64x64xf32>
    return %1 : f32
  }
}

