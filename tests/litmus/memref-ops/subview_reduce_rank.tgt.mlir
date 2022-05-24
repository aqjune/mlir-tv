func.func @subview(%arg: memref<8x16x4xf32>) -> f32 {
   %c1 = arith.constant 1 : index
   %1 = memref.subview %arg[0, 0, 0][1, 16, 4][1, 1, 1] : memref<8x16x4xf32> to memref<16x4xf32>
   %2 = memref.load %1[%c1, %c1]: memref<16x4xf32>
   return %2 : f32
 }
