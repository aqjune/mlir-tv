// VERIFY

func.func @fold_rank_reducing_subview_with_load
    (%arg0 : memref<?x?x?x?x?x?xf32>, %arg1 : index, %arg2 : index,
     %arg3 : index, %arg4 : index, %arg5 : index, %arg6 : index,
     %arg7 : index, %arg8 : index, %arg9 : index, %arg10: index,
     %arg11 : index, %arg12 : index, %arg13 : index, %arg14: index,
     %arg15 : index, %arg16 : index) -> f32 {
  %0 = memref.subview %arg0[%arg1, %arg2, %arg3, %arg4, %arg5, %arg6][4, 1, 1, 4, 1, 1][%arg7, %arg8, %arg9, %arg10, %arg11, %arg12] : memref<?x?x?x?x?x?xf32> to memref<4x1x4x1xf32, offset:?, strides: [?, ?, ?, ?]>
  %1 = memref.load %0[%arg13, %arg14, %arg15, %arg16] : memref<4x1x4x1xf32, offset:?, strides: [?, ?, ?, ?]>
  return %1 : f32
}

// How to reproduce tgt:
// mlir-opt -fold-memref-subview-ops <src>
