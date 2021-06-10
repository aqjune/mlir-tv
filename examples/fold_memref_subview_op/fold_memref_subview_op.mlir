func @fold_subview(%arg0: tensor<64x64xf32>, %i0: index) -> f32 {
    %0 = memref.buffer_cast %arg0 : memref<64x64xf32>
    %1 = memref.subview %0[0, 0][1, 64][1, 1]
    : memref <64x64xf32> to memref<64xf32>
    %2 = memref.load %1[%i0] : memref<64xf32>
    return %2 : f32
}