// VERIFY

 func @subview(%arg: memref<8x16x4xf32>) -> f32 {
   %c0 = arith.constant 0 : index
   %c1 = arith.constant 1 : index
   %1 = memref.load %arg[%c0, %c1, %c1]: memref<8x16x4xf32>
   return %1 : f32
 }
