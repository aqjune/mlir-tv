// VERIFY-INCORRECT

// Show that subview does not raise UB if the resulting memref is out-of-bounds
func.func @subview(%arg: memref<4x4xf32>) -> i32 {
  %c0 = arith.constant 0: i32
  %f = memref.subview %arg[4, 0][1, 1][1, 1] : memref<4x4xf32> to memref<1x1xf32, affine_map<(d0, d1) -> (d0 * 4 + d1 + 16)>>
  return %c0: i32
}
