// VERIFY

func.func @generalize_fill(%output: memref<20x20xf32>, %value : f32) {
  linalg.fill ins(%value: f32) outs(%output: memref<20x20xf32>)
  return
}

// How to reproduce tgt:
// mlir-opt -linalg-generalize-named-ops <src>
