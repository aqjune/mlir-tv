func @conv(%input: tensor<1x225x225x3xf32>, %filter: tensor<3x3x3x32xf32>,
           %output: tensor<1x112x112x32xf32>) -> tensor<1x112x112x32xf32> {
  %0 = memref.buffer_cast %input : memref<1x225x225x3xf32>
  %1 = memref.buffer_cast %filter : memref<3x3x3x32xf32>
  %2 = memref.alloc() : memref<1x112x112x32xf32>
  linalg.conv(%1, %0, %2) {dilations = [1, 1], strides = [2, 2]}
    : memref<3x3x3x32xf32>, memref<1x225x225x3xf32>, memref<1x112x112x32xf32>
  %3 = memref.tensor_load %2 : memref<1x112x112x32xf32>
  return %3 : tensor<1x112x112x32xf32>
}
