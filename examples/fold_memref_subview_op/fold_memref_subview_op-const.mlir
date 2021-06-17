// iree-opt --fold-memref-subview-ops %s
#map = affine_map<(d0, d1)[s0] -> (d0 * 64 + s0 + d1)>
func @fold_subview(%arg0: tensor<64x64xf32>, %off_y: index, %off_x: index, %dim_y: index, %dim_x: index, %idx_y: index, %idx_x: index) -> f32 {
    %0 = memref.buffer_cast %arg0 : memref<64x64xf32>
    %1 = memref.subview %0[%off_y, %off_x][%dim_y, %dim_x][1, 1]
    : memref <64x64xf32> to memref<?x?xf32, #map>
    %2 = memref.load %1[%idx_y, %idx_x]: memref<?x?xf32, #map>
    return %2 : f32
}