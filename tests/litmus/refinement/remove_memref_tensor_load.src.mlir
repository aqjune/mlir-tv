// VERIFY

func.func @test(%arg : memref<2x3xf32>) -> f32
{
  %val = arith.constant 1.000000e-03 : f32
  %tensor = bufferization.to_tensor %arg : memref<2x3xf32>
  return %val : f32
}
