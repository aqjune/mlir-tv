// VERIFY

func @generalize_fill(%output: memref<20x20xf32>, %value : f32) {
  linalg.fill(%value, %output): f32, memref<20x20xf32>
  return
}

// How to reproduce tgt:
// mlir-opt -linalg-generalize-named-ops <src>
