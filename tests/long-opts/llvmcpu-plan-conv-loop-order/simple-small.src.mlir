// VERIFY

func @conv(%filter: memref<3x3x1x1xf32>, %input: memref<1x3x3x1xf32>,
           %output: memref<1x1x1x1xf32>) {
  linalg.conv(%filter, %input, %output)
    : memref<3x3x1x1xf32>, memref<1x3x3x1xf32>, memref<1x1x1x1xf32>
  return
}

// How to reproduce tgt:
// iree-opt -iree-llvmcpu-plan-conv-loop-order <src>
