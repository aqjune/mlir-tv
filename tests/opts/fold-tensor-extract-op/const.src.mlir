// VERIFY

func @fold_tensor_extract(%arg0 : memref<2x3xf32>) -> f32
{
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %0 = bufferization.to_tensor %arg0 : memref<2x3xf32>
  %1 = tensor.extract %0[%c1, %c2] : tensor<2x3xf32>
  return %1 : f32
}

// How to reproduce tgt:
// iree-opt -iree-codegen-fold-tensor-extract-op <src>
