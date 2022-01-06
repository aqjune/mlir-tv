// VERIFY

func @f(%arg : memref<1x2xf32>) -> memref<1x2xf32> {
  %0 = memref.expand_shape %arg [[0, 1], [2]] : memref<1x2xf32> into memref<1x1x2xf32>
  %1 = memref.collapse_shape %0 [[0, 1], [2]] : memref<1x1x2xf32> into memref<1x2xf32>
  return %1: memref<1x2xf32>
}
