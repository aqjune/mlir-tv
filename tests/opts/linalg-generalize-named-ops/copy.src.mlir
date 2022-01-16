// VERIFY

func @copy(%m1: memref<?xf32>, %m2: memref<?xf32>)
{
  linalg.copy(%m1, %m2) : memref<?xf32>, memref<?xf32>
  return
}

// How to reproduce tgt:
// mlir-opt -linalg-generalize-named-ops <src>

