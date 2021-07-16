// VERIFY

func @fold_tensor_extract(%arg0 : memref<2x3xf32>, %arg1 : memref<2x3xf32>) -> f32
{
  %c1 = constant 1 : index
  %c2 = constant 2 : index
  %0 = memref.tensor_load %arg0 : memref<2x3xf32>
  %1 = tensor.extract %0[%c1, %c2] : tensor<2x3xf32>
  memref.store %1, %arg1[%c1, %c2] : memref<2x3xf32>
  return %1 : f32
}

// How to reproduce tgt:
// iree-opt -iree-codegen-fold-tensor-extract-op <src>
