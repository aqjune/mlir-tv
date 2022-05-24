// VERIFY-INCORRECT

func.func @fold_tensor_extract(%arg0 : memref<2x3xf32>, %arg1 : memref<2x3xf32>) -> f32
{
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %0 = bufferization.to_tensor %arg0 : memref<2x3xf32>
  %1 = tensor.extract %0[%c1, %c0] : tensor<2x3xf32>
  memref.store %1, %arg1[%c1, %c2] : memref<2x3xf32>
  return %1 : f32
}

// How to reproduce tgt:
// iree-opt -iree-codegen-fold-tensor-extract-op <src>
