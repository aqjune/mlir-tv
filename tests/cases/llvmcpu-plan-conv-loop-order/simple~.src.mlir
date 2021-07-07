// VERIFY-INCORRECT

func @conv(%filter: memref<3x3x3x32xf32>, %input: memref<1x225x225x3xf32>,
           %output: memref<1x112x112x32xf32>) {
  linalg.conv(%filter, %input, %output) {dilations = [1, 1], strides = [2, 2]}
    : memref<3x3x3x32xf32>, memref<1x225x225x3xf32>, memref<1x112x112x32xf32>
  return
}

// How to reproduce tgt:
// iree-opt -iree-llvmcpu-plan-conv-loop-order <src>
