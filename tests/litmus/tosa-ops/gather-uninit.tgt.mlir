func.func @test_gather() -> tensor<2x1x1xf32> {
  %v = linalg.init_tensor [2,1,1]: tensor<2x1x1xf32>
  return %v: tensor<2x1x1xf32>
}
