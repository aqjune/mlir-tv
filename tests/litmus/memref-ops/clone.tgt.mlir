func @f(%arg : memref<2x3xf32>) -> tensor<2x3xf32>
{
  %v = memref.clone %arg: memref<2x3xf32> to memref<2x3xf32>
  %0 = memref.tensor_load %v : memref<2x3xf32>
  return %0: tensor<2x3xf32>
}
