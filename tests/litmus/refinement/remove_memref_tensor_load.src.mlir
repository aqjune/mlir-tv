// VERIFY

func @test(%arg : memref<2x3xf32>) -> f32
{
  %val = constant 1.000000e-03 : f32
  %tensor = memref.tensor_load %arg : memref<2x3xf32>
  return %val : f32
}
