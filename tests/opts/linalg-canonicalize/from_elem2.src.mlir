// VERIFY

func.func @from_elem2()
    -> (f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32) {
  %f0 = arith.constant 0.0 : f32
  %f1 = arith.constant 1.0 : f32
  %f2 = arith.constant 2.0 : f32
  %f3 = arith.constant 3.0 : f32
  %f4 = arith.constant 4.0 : f32
  %f5 = arith.constant 5.0 : f32
  %f6 = arith.constant 6.0 : f32
  %f7 = arith.constant 7.0 : f32
  %f8 = arith.constant 8.0 : f32
  %f9 = arith.constant 9.0 : f32
  %f10 = arith.constant 10.0 : f32
  %f11 = arith.constant 11.0 : f32
  %tensor = tensor.from_elements %f0,%f1,%f2,%f3,%f4,%f5,%f6,%f7,%f8,%f9,%f10,%f11
         : tensor<3x2x2xf32>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %r0 = tensor.extract %tensor[%c0, %c0, %c0] : tensor<3x2x2xf32>
  %r1 = tensor.extract %tensor[%c0, %c0, %c1] : tensor<3x2x2xf32>
  %r2 = tensor.extract %tensor[%c0, %c1, %c0] : tensor<3x2x2xf32>
  %r3 = tensor.extract %tensor[%c0, %c1, %c1] : tensor<3x2x2xf32>
  %r4 = tensor.extract %tensor[%c1, %c0, %c0] : tensor<3x2x2xf32>
  %r5 = tensor.extract %tensor[%c1, %c0, %c1] : tensor<3x2x2xf32>
  %r6 = tensor.extract %tensor[%c1, %c1, %c0] : tensor<3x2x2xf32>
  %r7 = tensor.extract %tensor[%c1, %c1, %c1] : tensor<3x2x2xf32>
  %r8 = tensor.extract %tensor[%c2, %c0, %c0] : tensor<3x2x2xf32>
  %r9 = tensor.extract %tensor[%c2, %c0, %c1] : tensor<3x2x2xf32>
  %r10 = tensor.extract %tensor[%c2, %c1, %c0] : tensor<3x2x2xf32>
  %r11 = tensor.extract %tensor[%c2, %c1, %c1] : tensor<3x2x2xf32>
  return %r0,%r1,%r2,%r3,%r4,%r5,%r6,%r7,%r8,%r9,%r10,%r11
         : f32,f32,f32,f32,f32,f32,f32,f32,f32,f32,f32,f32
}

// How to reproduce tgt:
// mlir-opt -canonicalize <src>