// EXPECT: "correct (source is always undefined)"

func @fold_rank_reducing_subview_with_load(%arg0 : memref<?x?xf32>) -> f32 {
  %c0 = constant 0: index
  %c1 = constant 1: index
  %ptr = memref.subview %arg0[%c0, %c0][1, 1][%c1, %c1] : memref<?x?xf32> to memref<1xf32, offset:?, strides: [?]>
  %1 = memref.load %ptr[%c1] : memref<1xf32, offset:?, strides: [?]>
  return %1 : f32
}
