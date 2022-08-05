// VERIFY-INCORRECT

#map = affine_map<(d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>
func @fold_subview(%arg0: tensor<?x?xf32>, %off_y: index, %off_x: index, %dim_y: index, %dim_x: index, %idx_y: index, %idx_x: index) -> f32 {
    %0 = bufferization.to_memref %arg0 : memref<?x?xf32>
    %1 = memref.subview %0[%off_y, %off_x][%dim_y, %dim_x][1, 1]
    : memref <?x?xf32> to memref<?x?xf32, #map>
    %2 = memref.load %1[%idx_y, %idx_x]: memref<?x?xf32, #map>
    return %2 : f32
}
