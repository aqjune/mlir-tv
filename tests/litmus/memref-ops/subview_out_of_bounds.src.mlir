// EXPECT: "correct (source is always undefined)"

func.func @fold_rank_reducing_subview_with_load(%arg0 : memref<?x?xf32>) -> f32 {
  %c0 = arith.constant 0: index
  %c1 = arith.constant 1: index
  %ptr = memref.subview %arg0[%c0, %c0][1, 1][%c1, %c1] : memref<?x?xf32> to memref<1xf32, strided<[?], offset: ?>>
  %1 = memref.load %ptr[%c1] : memref<1xf32, strided<[?], offset: ?>>
  return %1 : f32
}
