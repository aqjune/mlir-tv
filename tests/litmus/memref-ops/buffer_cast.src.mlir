// EXPECT: "Memory mismatch"

// Although this conversion is correct, currently we don't fully support local block memory refinement.
// So this test just checks whether return value mismatch.

func @buffer_cast(%arg : tensor<2x3xf32>) -> f32
{
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %0 = memref.buffer_cast %arg : memref<2x3xf32>
  %1 = memref.load %0[%c0, %c1] : memref<2x3xf32>
  return %1 : f32
}
