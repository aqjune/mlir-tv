
func @fold_rank_reducing_subview_with_load(%arg0: memref<?x?xf32>) -> f32 {
    %c1 = constant 1 : index
    %c0 = constant 0 : index
    %0 = memref.load %arg0[%c1, %c0] : memref<?x?xf32>
    return %0 : f32
}
