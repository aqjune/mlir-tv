func.func @sum(%x: tensor<1xf32>) -> f32
 {
   %c0 = arith.constant 0 : index
   %elem = tensor.extract %x[%c0]: tensor<1xf32>
   return %elem : f32
 }
