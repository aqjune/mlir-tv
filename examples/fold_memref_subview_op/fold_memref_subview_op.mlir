func @fold_subview(%arg0: tensor<64x64xf32>, %iy: index, %ix: index) -> f32 {
    %i0 = constant 0: index
    %0 = memref.buffer_cast %arg0 : memref<64x64xf32>
    %1 = memref.subview %0[%iy, 0][1, 64][1, 1]
    : memref <64x64xf32> to memref<1x64xf32, affine_map<(d0, d1)[s0] -> (d0 * 64 + s0 + d1)>>
    %2 = memref.load %1[%i0, %ix]: memref<1x64xf32, affine_map<(d0, d1)[s0] -> (d0 * 64 + s0 + d1)>>
    return %2 : f32
}