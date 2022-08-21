func @test(%arg0: tensor<2x5x5x2xf32>) -> tensor<2x7x7x2xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = linalg.pad_tensor %arg0 low[0, 1, 1, 0] high[0, 1, 1, 0]  {
  ^bb0(%arg3: index, %arg4: index, %arg5: index, %arg6: index):  // no predecessors
    linalg.yield %cst : f32
  } : tensor<2x5x5x2xf32> to tensor<2x7x7x2xf32>
  return %0 : tensor<2x7x7x2xf32>
}
