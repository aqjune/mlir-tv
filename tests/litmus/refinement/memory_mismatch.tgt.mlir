func @test(%arg0 : memref<2x3xf32>) -> f32
{
  %index = constant 0 : index
  %val = constant 1.000000e-03 : f32
  return %val : f32
}
